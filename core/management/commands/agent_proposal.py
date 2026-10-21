"""Run one AI proposal cycle from the command line.

Deterministic fallback unless OPENROUTER_API_KEY is set. Useful for the
A/B/C evaluation harness and for scripting demo state.
"""

from django.core.management.base import BaseCommand

from ... import agents
from ...models import AgentRun


class Command(BaseCommand):
    help = 'Run the Inventory Analyst + Distribution Planner once.'

    def add_arguments(self, parser):
        parser.add_argument('--username', default=None,
                            help='Coordinator username to own the drafts')

    def handle(self, *args, **options):
        from django.contrib.auth.models import User
        username = options['username']
        if username:
            user = User.objects.get(username=username)
        else:
            user = (User.objects.filter(profile__role='COORDINATOR').first()
                    or User.objects.filter(is_superuser=True).first())
        analyst_run, planner_run = agents.run_proposal_cycle(user)
        self.stdout.write(
            f'Analyst run #{analyst_run.pk}: {analyst_run.status} '
            f'({analyst_run.latency_ms} ms, {len(analyst_run.tool_calls)} tool calls)')
        self.stdout.write(
            f'Planner run #{planner_run.pk}: {planner_run.status} '
            f'(plan={planner_run.plan_id}, {planner_run.latency_ms} ms, '
            f'{len(planner_run.tool_calls)} tool calls)')
