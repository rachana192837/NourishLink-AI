"""Minimal OpenRouter chat-completions client (stdlib only, no dependencies).

OpenRouter normalises providers behind an OpenAI-compatible chat API:
POST https://openrouter.ai/api/v1/chat/completions with a Bearer key.
Free models carry a ``:free`` suffix and ``openrouter/free`` is an
auto-router that selects a free model which supports the features the
request needs (e.g. tools). The served model id is returned in the
response body and recorded on AgentRun for evaluation.

Every failure mode (missing key, HTTP 402/408/429/5xx, network, malformed
payload) raises LLMUnavailable so callers can fall back deterministically.
"""

import json
import os
import urllib.error
import urllib.request

BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Errors that mean "try the deterministic fallback", not "crash".
FALLBACK_HTTP_CODES = {402, 408, 429, 500, 502, 503, 504}


class LLMUnavailable(Exception):
    """No LLM response is available; callers should fall back."""


def api_key():
    return os.environ.get('OPENROUTER_API_KEY', '').strip()


def model_name():
    return os.environ.get('OPENROUTER_MODEL', 'openrouter/free')


def vision_model_name():
    return os.environ.get('OPENROUTER_VISION_MODEL',
                          'nvidia/nemotron-3.5-super-vl:free')


# Free vision models churn on OpenRouter (rate limits, catalog edits), so
# chat_vision walks this chain: primary = env-configured/default, then
# verified alternates. Reasoning-style VL models can return an empty
# completion when max_tokens is too small for their hidden reasoning —
# the retry escalates the token budget before giving up on a model.
VISION_MODEL_FALLBACKS = [
    'google/gemma-4-31b-it:free',
    'google/gemma-4-26b-a4b-it:free',
]


def chat(messages, tools=None, tool_choice=None, temperature=0.2,
         max_tokens=1200, timeout=120, model=None):
    """One chat-completion round-trip.

    Returns {'content', 'tool_calls', 'model', 'prompt_tokens',
    'completion_tokens'}. Raises LLMUnavailable when the LLM cannot be
    reached or answers unusably. ``messages`` content may be a plain
    string or a multimodal part list (text + image_url) for vision calls;
    pass ``model`` explicitly for those (the auto-router does not
    reliably pick a vision-capable model).
    """
    key = api_key()
    if not key:
        raise LLMUnavailable('OPENROUTER_API_KEY is not set.')

    body = {
        'model': model or model_name(),
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if tools:
        body['tools'] = tools
    if tool_choice:
        body['tool_choice'] = tool_choice

    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:8000/',
            'X-Title': 'NourishLink AI',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8', errors='replace')[:300]
        except Exception:
            pass
        if exc.code in FALLBACK_HTTP_CODES:
            raise LLMUnavailable(f'HTTP {exc.code}: {detail}') from exc
        raise LLMUnavailable(f'HTTP {exc.code}: {detail}') from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMUnavailable(f'Network error: {exc}') from exc
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LLMUnavailable(f'Unparseable response: {exc}') from exc

    try:
        choice = payload['choices'][0]['message']
        usage = payload.get('usage') or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f'Unexpected response shape: {payload!r:.300}') from exc

    content = choice.get('content')
    tool_calls = choice.get('tool_calls') or []

    if content in (None, '') and not tool_calls:
        raise LLMUnavailable('Empty completion (no content, no tool calls).')

    prompt_tokens = usage.get('prompt_tokens') or _estimate_tokens(messages)
    completion_tokens = usage.get('completion_tokens') or _estimate_tokens(
        [{'content': content or ''}]) + sum(
        _estimate_tokens([{'content': tc.get('function', {}).get('arguments', '')}])
        for tc in tool_calls)

    return {
        'content': content,
        'tool_calls': tool_calls,
        'model': payload.get('model', model_name()),
        'prompt_tokens': int(prompt_tokens),
        'completion_tokens': int(completion_tokens),
    }


def chat_vision(prompt, image_bytes, mime='image/png',
                temperature=0.1, max_tokens=900, timeout=120):
    """One vision round-trip: text prompt + one image (raw bytes).

    Walks the vision model chain (env override first, then verified free
    alternates) and retries once with a larger token budget on empty
    completions (reasoning models need headroom). Returns the same dict
    shape as chat(); raises LLMUnavailable when every attempt fails.
    """
    import base64
    import time

    data_uri = f'data:{mime};base64,{base64.b64encode(image_bytes).decode("ascii")}'
    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': data_uri}},
        ]},
    ]

    chain = [vision_model_name()]
    for candidate in VISION_MODEL_FALLBACKS:
        if candidate not in chain:
            chain.append(candidate)

    last_error = None
    for chain_idx, model_id in enumerate(chain):
        for attempt, tokens in enumerate((max_tokens, max_tokens * 4)):
            try:
                return chat(messages, temperature=temperature,
                            max_tokens=tokens, timeout=timeout,
                            model=model_id)
            except LLMUnavailable as exc:
                last_error = exc
                message = str(exc)
                if 'Empty completion' not in message:
                    break  # other failures: move to the next model
                if attempt == 0 and 'Empty completion' in message:
                    time.sleep(2)  # retry same model with 4x budget
        if chain_idx < len(chain) - 1:
            time.sleep(8)  # brief cool-down before the next model
    raise last_error


def _estimate_tokens(messages):
    # ~4 characters per token is the usual English ballpark; good enough
    # when a free provider omits usage accounting.
    total = 0
    for m in messages:
        content = m.get('content')
        if isinstance(content, str):
            total += len(content)
        elif content:  # multimodal part list — estimate from its JSON
            total += len(json.dumps(content, default=str))
    return max(1, total // 4)
