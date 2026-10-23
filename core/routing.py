"""Delivery-route grouping for the dispatch board.

Groups a plan's allocation items into numbered truck routes from the
warehouse (depot), using a greedy cheapest-insertion heuristic. The
output is presentation data only — nothing is persisted by this module
(the dispatch view stores the returned record on the plan).

Road geometry: when the public OSRM demo router is reachable, each truck
route is upgraded to real driving polylines with per-stop ETA minutes;
any network failure degrades gracefully to indicative straight lines
(`road_route: False`), the same offline-first principle as the app's
map tiles.

Deterministic except for the OSRM-updated distances: same plan + same
truck count → same stop ORDER always; km values may differ slightly if
the router is unavailable.
"""

import json
import math
import urllib.request

# Peenya Industrial Area — the synthetic food-bank warehouse ("depot"),
# in Bengaluru's warehouse belt, west of the partner localities.
DEPOT = (13.0284, 77.5199)

TRUCK_COLORS = ['#e74c3c', '#2980b9', '#8e44ad', '#27ae60']

OSRM_URL = ('https://router.project-osrm.org/route/v1/driving/'
            '{coords}?overview=full&geometries=geojson&steps=false')
OSRM_TIMEOUT_S = 3

# Per-process memo: after one OSRM failure, stop retrying until restart
# (or a reset) — keeps the dispatch view fast when offline.
_osrm_down = False


def reset_osrm_state():
    """Clear the reachability memo (used by tests)."""
    global _osrm_down
    _osrm_down = False


def haversine_km(p1, p2):
    lat1, lng1, lat2, lng2 = p1[0], p1[1], p2[0], p2[1]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _distance(a, b):
    return haversine_km(a, b)


def _osrm_route(points, timeout_s=OSRM_TIMEOUT_S):
    """Real driving route through points [(lat, lng), ...] via OSRM.

    Returns {'polyline': [(lat, lng), ...], 'distance_km': float,
             'duration_min': float, 'leg_minutes': [float, ...]} —
    one leg between consecutive waypoints. Raises on any problem.
    """
    coords = ';'.join(f'{p[1]:.6f},{p[0]:.6f}' for p in points)
    with urllib.request.urlopen(OSRM_URL.format(coords=coords),
                                timeout=timeout_s) as response:
        payload = json.loads(response.read().decode('utf-8'))
    route = payload['routes'][0]
    polyline = [(coord[1], coord[0]) for coord in route['geometry']['coordinates']]
    leg_minutes = [leg['duration'] / 60.0 for leg in route['legs']]
    return {'polyline': polyline,
            'distance_km': route['distance'] / 1000.0,
            'duration_min': route['duration'] / 60.0,
            'leg_minutes': leg_minutes}


def _enrich_truck_route(truck, depot, points):
    """Attach road polylines + per-stop ETA to one truck route.

    On OSRM failure (or offline), trucks keep `distance_km` straight-line
    and get `road_route: False`; stops keep no eta. Deterministic fallback.
    """
    global _osrm_down
    truck['road_route'] = False
    if not truck['stops']:
        return truck
    if _osrm_down:
        return truck
    try:
        legs = _osrm_route([depot] + points, timeout_s=OSRM_TIMEOUT_S)
    except Exception:
        # Router unreachable → render straight lines; don't stall the
        # remaining trucks with repeated timeouts.
        _osrm_down = True
        return truck
    truck['road_route'] = True
    truck['polyline'] = [list(p) for p in legs['polyline']]
    truck['distance_km'] = round(legs['distance_km'], 1)
    cumulative = 0.0
    for idx, stop in enumerate(truck['stops']):
        # legs: one per waypoint pair; leg 0 = depot → stop 1.
        cumulative += legs['leg_minutes'][idx]
        stop['eta_min'] = round(cumulative)
    return truck


def group_plan_into_trucks(plan, deliveries, n_trucks=2):
    """Compute truck routes for one allocation plan.

    Args:
        plan: AllocationPlan instance (for id/labels only).
        deliveries: list of dicts, one per destination organisation:
            {'org_id', 'org_name', 'lat', 'lng', 'units',
             'items': [{'product', 'qty', 'batch_code'}]}
    Returns:
        {'plan_id', 'plan_name', 'depot': (lat, lng),
         'trucks': [{'label', 'color', 'stops': [...], 'total_units',
                     'distance_km'}],
         'unmapped': [org names without coordinates]}
    """
    depot = DEPOT
    mapped = []
    unmapped = []
    for stop in deliveries:
        if stop.get('lat') is None or stop.get('lng') is None:
            unmapped.append(stop['org_name'])
        else:
            stop['pos'] = (stop['lat'], stop['lng'])
            mapped.append(stop)

    n_trucks = max(1, min(int(n_trucks), len(TRUCK_COLORS)))
    empty = [{'label': f'Truck {i + 1}',
              'color': TRUCK_COLORS[i],
              'stops': [], 'total_units': 0, 'distance_km': 0}
             for i in range(n_trucks)]

    if not mapped:
        return {'plan_id': plan.pk, 'plan_name': plan.name, 'depot': depot,
                'trucks': empty, 'unmapped': unmapped, 'road_route': False}

    mapped.sort(key=lambda s: -s['units'])
    routes = [[] for _ in range(n_trucks)]
    trucks = []

    def route_len(route):
        pts = [depot] + [s['pos'] for s in route] + [depot]
        return sum(_distance(pts[i], pts[i + 1])
                   for i in range(len(pts) - 1))

    for stop in mapped:
        best = None
        for t_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                candidate = route[:pos] + [stop] + route[pos:]
                cost = route_len(candidate)
                if best is None or cost < best[0]:
                    best = (cost, t_idx, candidate)
        routes[best[1]] = best[2]

    for i, route in enumerate(routes):
        seq_stops = [dict(stop, seq=j + 1)
                     for j, stop in enumerate(route)]
        total_units = sum(stop['units'] for stop in seq_stops)
        trucks.append({
            'label': f'Truck {i + 1}',
            'color': TRUCK_COLORS[i],
            'stops': seq_stops,
            'total_units': total_units,
            'distance_km': round(route_len(route), 1),
        })
        trucks[i] = _enrich_truck_route(
            trucks[i], depot, [stop['pos'] for stop in seq_stops])

    return {'plan_id': plan.pk, 'plan_name': plan.name, 'depot': depot,
            'trucks': trucks, 'unmapped': unmapped,
            'road_route': any(t.get('road_route') for t in trucks)}
