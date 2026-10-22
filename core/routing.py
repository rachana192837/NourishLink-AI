"""Delivery-route grouping for the dispatch board.

Groups a plan's allocation items into numbered truck routes from the
warehouse (depot), using a greedy cheapest-insertion heuristic. The
output is presentation data only — nothing is persisted, and the
routes are indicative straight-line sequences, not GPS-planned roads.

Deterministic: same plan + same truck count → same routes.
"""

import math

# Bengaluru city centre — the synthetic food-bank warehouse ("depot").
DEPOT = (12.9716, 77.5946)

TRUCK_COLORS = ['#e74c3c', '#2980b9', '#8e44ad', '#27ae60']


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
                'trucks': empty, 'unmapped': unmapped}

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

    return {'plan_id': plan.pk, 'plan_name': plan.name, 'depot': depot,
            'trucks': trucks, 'unmapped': unmapped}
