#!/usr/bin/env python3
"""
UK Sports Club Bike Tour - Route Optimizer v3
Uses OSRM road distances + region-constrained TSP.
Start: Swansea (Ospreys). End: Aberdeen.
"""

import json
import math
import os
import time
import urllib.request
import urllib.parse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

MAX_DAILY_MILES = 90.0
CYCLING_FACTOR = 1.10  # driving → cycling correction
OSRM_BASE = "https://router.project-osrm.org"
OSRM_PROFILE = "driving"
CACHE_FILE = "distance_matrix_cache.npz"
START_VENUE = "Ospreys"  # Swansea

# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Venue:
    name: str
    category: str
    postcode: str
    address: str
    lat: float
    lon: float
    region: str
    country: str
    loc_id: int = -1        # index into unique locations
    tour_region: str = ""   # assigned tour region

# ─── Region Assignment ───────────────────────────────────────────────────────

def assign_tour_region(v: Venue) -> str:
    """Assign venue to a tour region based on geography."""
    lat, lon = v.lat, v.lon
    name = v.name

    # Scotland
    if v.country == "Scotland" or v.region == "Scotland":
        return "Scotland"

    # Wales — Wrexham goes with North_West (geographically on the NW corridor)
    if v.country == "Wales" or v.region == "Wales":
        if "Wrexham" in name:
            return "North_West"
        return "South_Wales"

    # North East
    if v.region == "North East":
        return "North_East"

    # Yorkshire
    if v.region == "Yorkshire and The Humber":
        if lat < 53.55 and lon > -0.5:
            return "South_Yorkshire_Humber"
        if lat < 53.6:
            return "South_Yorkshire_Humber"
        return "West_Yorkshire"

    # North West — also include Stoke/Port Vale/Shrewsbury (NW corridor)
    if v.region == "North West":
        return "North_West"

    # West Midlands — Stoke, Port Vale, Shrewsbury go to North_West
    if v.region == "West Midlands":
        if any(x in name for x in ["Stoke", "Port Vale", "Shrewsbury"]):
            return "North_West"
        return "West_Midlands"

    # East Midlands — Northampton clubs go to Thames_Valley (between Peterborough and MK)
    if v.region == "East Midlands":
        if "Northampton" in name:
            return "Thames_Valley"
        return "East_Midlands"

    # London
    if v.region == "London":
        return "London"

    # East of England
    if v.region == "East of England":
        if lat >= 52.0:
            return "East_Anglia"
        if lon > 0.3:
            return "East_Anglia"  # Colchester, Essex
        return "Home_Counties"

    # South East — fix Brighton/Crawley into Sussex_Kent, Hampshire into South_Coast
    if v.region == "South East":
        # Name-based overrides first
        if any(x in name for x in ["Brighton", "Crawley", "Horsham"]):
            return "Sussex_Kent"
        if any(x in name for x in ["Portsmouth", "Southampton", "Hampshire", "Chichester"]):
            return "South_Coast"
        if lon > 0.3:
            return "Sussex_Kent"  # Kent venues (Gillingham, Canterbury)
        if lat < 51.15 and lon > -0.7:
            return "Sussex_Kent"  # Catch remaining Sussex
        if lat < 51.3 and lon > -0.5:
            return "Sussex_Kent"  # Bromley area
        # Thames Valley: Reading, Oxford, Wycombe, MK, Guildford, Surrey
        return "Thames_Valley"

    # South West — Bournemouth to South_Coast, Swindon to Thames_Valley
    if v.region == "South West":
        if "Bournemouth" in name:
            return "South_Coast"
        if "Swindon" in name:
            return "Thames_Valley"
        if lat < 51.1:
            return "Devon_Somerset"
        if any(x in name for x in ["Gloucester", "Cheltenham"]):
            return "Bristol_Bath_Gloucester"
        return "Bristol_Bath_Gloucester"

    # Fallback
    return "Other"


# ─── Hardcoded Regional Order ────────────────────────────────────────────────

REGION_ORDER = [
    "South_Wales",
    "Bristol_Bath_Gloucester",
    "Devon_Somerset",
    "South_Coast",
    "Sussex_Kent",
    "London",
    "Home_Counties",
    "East_Anglia",
    "Thames_Valley",
    "West_Midlands",
    "East_Midlands",
    "South_Yorkshire_Humber",
    "West_Yorkshire",
    "North_West",
    "North_East",
    "Scotland",
]

# Preferred entry/exit for each region (venue names)
# Entry = where we come in from previous region, Exit = where we leave to next
REGION_HINTS = {
    "South_Wales": {"entry": "Ospreys", "exit": "Newport County"},
    "Bristol_Bath_Gloucester": {"entry": "Gloucester Rugby", "exit": "Somerset"},
    "Devon_Somerset": {"entry": "Somerset", "exit": "Somerset"},
    "South_Coast": {"entry": "Bournemouth", "exit": "Hampshire"},
    "Sussex_Kent": {"entry": "Brighton & Hove Albion", "exit": "Kent"},
    "London": {"entry": "Charlton Athletic", "exit": "Saracens"},
    "Home_Counties": {"entry": "Watford", "exit": "Stevenage"},
    "East_Anglia": {"entry": "Cambridge United", "exit": "Peterborough United"},
    "Thames_Valley": {"entry": "Northampton Town", "exit": "Swindon Town"},
    "West_Midlands": {"entry": "Cheltenham Town", "exit": "Coventry City"},
    "East_Midlands": {"entry": "Coventry City", "exit": "Chesterfield"},
    "South_Yorkshire_Humber": {"entry": "Sheffield United", "exit": "Hull Kingston Rovers"},
    "West_Yorkshire": {"entry": "Castleford Tigers", "exit": "Huddersfield Town"},
    "North_West": {"entry": "Manchester City", "exit": "Barrow"},
    "North_East": {"entry": "Middlesbrough", "exit": "Newcastle Red Bulls"},
    "Scotland": {"entry": "Kilmarnock", "exit": "Aberdeen"},
}

# ─── Venue Loading ───────────────────────────────────────────────────────────

def load_venues(path: str) -> list[Venue]:
    with open(path) as f:
        data = json.load(f)

    venues = []
    for v in data:
        # Skip Northern Ireland
        if v.get("country") == "Northern Ireland":
            continue
        venue = Venue(
            name=v["name"],
            category=v["category"],
            postcode=v["postcode"],
            address=v["address"],
            lat=v["lat"],
            lon=v["lon"],
            region=v.get("region", ""),
            country=v.get("country", ""),
        )
        venue.tour_region = assign_tour_region(venue)
        venues.append(venue)

    return venues


def build_unique_locations(venues: list[Venue]) -> tuple[list[tuple[float, float]], dict[int, list[Venue]]]:
    """Group venues by physical location (rounded coords). Returns (locations, loc_to_venues)."""
    loc_map = {}  # (lat4, lon4) -> loc_id
    locations = []
    loc_to_venues = {}

    for v in venues:
        key = (round(v.lat, 4), round(v.lon, 4))
        if key not in loc_map:
            loc_id = len(locations)
            loc_map[key] = loc_id
            locations.append((v.lat, v.lon))
            loc_to_venues[loc_id] = []
        v.loc_id = loc_map[key]
        loc_to_venues[v.loc_id].append(v)

    return locations, loc_to_venues


# ─── OSRM Distance Matrix ───────────────────────────────────────────────────

def osrm_table(locations: list[tuple[float, float]], sources: list[int], destinations: list[int]) -> list[list[float]]:
    """Query OSRM Table API. Returns distances in meters."""
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in locations)
    src_str = ";".join(str(s) for s in sources)
    dst_str = ";".join(str(d) for d in destinations)

    url = f"{OSRM_BASE}/table/v1/{OSRM_PROFILE}/{coord_str}?sources={src_str}&destinations={dst_str}&annotations=distance"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BikeRoute/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            if data.get("code") == "Ok":
                return data["distances"]
            print(f"  OSRM error: {data.get('code')}, retrying...")
        except Exception as e:
            print(f"  Request failed (attempt {attempt+1}): {e}")
        time.sleep(2 ** attempt)

    raise RuntimeError("OSRM table request failed after 3 attempts")


def build_distance_matrix(locations: list[tuple[float, float]]) -> np.ndarray:
    """Build NxN road distance matrix in miles via OSRM, with caching."""
    n = len(locations)

    # Check cache
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE)
        if "matrix" in cached and cached["matrix"].shape == (n, n):
            locs = cached.get("locations", None)
            if locs is not None and locs.shape[0] == n:
                # Verify locations match
                current = np.array(locations)
                if np.allclose(locs, current, atol=0.0001):
                    print(f"Using cached distance matrix ({n}x{n})")
                    return cached["matrix"]
        print("Cache exists but doesn't match, rebuilding...")

    print(f"Building {n}x{n} road distance matrix via OSRM...")
    matrix = np.zeros((n, n))

    # Batch: max 10000 elements per request
    batch_size = min(n, 10000 // n)
    total_batches = math.ceil(n / batch_size)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        sources = list(range(start, end))
        destinations = list(range(n))

        print(f"  Batch {batch_idx+1}/{total_batches}: sources {start}-{end-1}")
        distances = osrm_table(locations, sources, destinations)

        for i, src in enumerate(sources):
            for j, dst in enumerate(destinations):
                val = distances[i][j]
                if val is None:
                    # Fallback: haversine
                    val = haversine_miles(locations[src], locations[dst]) * 1.4 * 1609.34
                matrix[src][dst] = val

        if batch_idx < total_batches - 1:
            time.sleep(1.5)  # rate limiting

    # Convert meters to miles, apply cycling factor
    matrix = matrix / 1609.34 * CYCLING_FACTOR

    # Cache
    np.savez(CACHE_FILE, matrix=matrix, locations=np.array(locations))
    print(f"Distance matrix cached to {CACHE_FILE}")

    return matrix


def haversine_miles(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Haversine distance in miles."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


# ─── TSP Solvers ─────────────────────────────────────────────────────────────

def nearest_neighbor(dist: np.ndarray, nodes: list[int], start: int, end: Optional[int] = None) -> list[int]:
    """Greedy nearest-neighbor tour. Fixed start, optionally fixed end."""
    if len(nodes) <= 1:
        return list(nodes)

    remaining = set(nodes)
    remaining.discard(start)
    if end is not None:
        remaining.discard(end)

    tour = [start]
    current = start

    while remaining:
        nearest = min(remaining, key=lambda x: dist[current][x])
        tour.append(nearest)
        remaining.remove(nearest)
        current = nearest

    if end is not None and end != start:
        tour.append(end)

    return tour


def tour_cost(dist: np.ndarray, tour: list[int]) -> float:
    """Total distance of a tour."""
    return sum(dist[tour[i]][tour[i+1]] for i in range(len(tour) - 1))


def two_opt(dist: np.ndarray, tour: list[int], fixed_start: bool = True, fixed_end: bool = False) -> list[int]:
    """2-opt local search improvement."""
    n = len(tour)
    if n < 4:
        return tour

    improved = True
    best = list(tour)

    while improved:
        improved = False
        lo = 1 if fixed_start else 0
        hi = n - 1 if fixed_end else n

        for i in range(lo, hi - 1):
            for j in range(i + 1, hi):
                # Cost of reversing segment [i..j]
                old_cost = 0
                new_cost = 0

                if i > 0:
                    old_cost += dist[best[i-1]][best[i]]
                    new_cost += dist[best[i-1]][best[j]]
                if j < n - 1:
                    old_cost += dist[best[j]][best[j+1]]
                    new_cost += dist[best[i]][best[j+1]]

                if new_cost < old_cost - 0.01:
                    best[i:j+1] = reversed(best[i:j+1])
                    improved = True

    return best


def or_opt(dist: np.ndarray, tour: list[int], fixed_start: bool = True, fixed_end: bool = False) -> list[int]:
    """Or-opt: move segments of 1, 2, or 3 nodes to better positions."""
    n = len(tour)
    if n < 4:
        return tour

    improved = True
    best = list(tour)

    while improved:
        improved = False
        lo = 1 if fixed_start else 0
        hi = n - (1 if fixed_end else 0)

        for seg_len in [1, 2, 3]:
            for i in range(lo, hi - seg_len + 1):
                seg = best[i:i+seg_len]
                before = best[:i] + best[i+seg_len:]

                for j in range(lo, len(before)):
                    if j == i:
                        continue
                    candidate = before[:j] + seg + before[j:]

                    if tour_cost(dist, candidate) < tour_cost(dist, best) - 0.01:
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best


def solve_tsp(dist: np.ndarray, nodes: list[int], start: int, end: Optional[int] = None) -> list[int]:
    """Full TSP solve: nearest-neighbor + 2-opt + or-opt."""
    if len(nodes) <= 2:
        if end is not None and end in nodes and end != start:
            return [start, end] if start in nodes else list(nodes)
        return [start] if start in nodes else list(nodes)

    tour = nearest_neighbor(dist, nodes, start, end)
    has_end = end is not None and end != start

    # Iterate 2-opt + or-opt until stable
    for _ in range(10):
        old_cost = tour_cost(dist, tour)
        tour = two_opt(dist, tour, fixed_start=True, fixed_end=has_end)
        tour = or_opt(dist, tour, fixed_start=True, fixed_end=has_end)
        if tour_cost(dist, tour) >= old_cost - 0.1:
            break

    return tour


# ─── Route Builder ───────────────────────────────────────────────────────────

def find_venue_loc(venues: list[Venue], name: str) -> Optional[int]:
    """Find location ID by venue name."""
    for v in venues:
        if v.name == name:
            return v.loc_id
    return None


def build_route(venues: list[Venue], locations: list[tuple[float, float]],
                loc_to_venues: dict[int, list[Venue]], dist: np.ndarray) -> list[Venue]:
    """Build complete ordered venue list using regional TSP."""

    # Group venues by tour region
    region_venues: dict[str, list[Venue]] = {}
    for v in venues:
        region_venues.setdefault(v.tour_region, []).append(v)

    # Check for unassigned
    for region, rvs in region_venues.items():
        if region not in REGION_ORDER and region != "Other":
            print(f"WARNING: Region '{region}' has {len(rvs)} venues but is not in REGION_ORDER")

    ordered_venues = []
    prev_exit_loc = find_venue_loc(venues, START_VENUE)
    visited_locs = set()

    for region_name in REGION_ORDER:
        rvs = region_venues.get(region_name, [])
        if not rvs:
            print(f"  Skipping empty region: {region_name}")
            continue

        # Get unique location IDs for this region
        region_locs = list(set(v.loc_id for v in rvs))

        # Determine entry/exit
        hints = REGION_HINTS.get(region_name, {})
        entry_name = hints.get("entry")
        exit_name = hints.get("exit")

        entry_loc = find_venue_loc(venues, entry_name) if entry_name else None
        exit_loc = find_venue_loc(venues, exit_name) if exit_name else None

        # If entry not in this region, find closest venue in region to prev_exit
        if entry_loc is None or entry_loc not in region_locs:
            if prev_exit_loc is not None:
                entry_loc = min(region_locs, key=lambda x: dist[prev_exit_loc][x])
            else:
                entry_loc = region_locs[0]

        if exit_loc is not None and exit_loc not in region_locs:
            exit_loc = None

        # Solve TSP within region
        tour_locs = solve_tsp(dist, region_locs, entry_loc, exit_loc)

        print(f"  {region_name}: {len(rvs)} venues, {len(tour_locs)} locations, "
              f"{tour_cost(dist, tour_locs):.1f} mi internal")

        # Convert location tour to venue list
        for loc_id in tour_locs:
            if loc_id in visited_locs:
                # Already visited (shared location across regions - shouldn't happen often)
                continue
            visited_locs.add(loc_id)
            # Add all venues at this location
            loc_venues = [v for v in rvs if v.loc_id == loc_id]
            if not loc_venues:
                # Venue might be from a different region sharing this location
                loc_venues = [v for v in loc_to_venues[loc_id] if v not in ordered_venues]
            ordered_venues.extend(loc_venues)

        prev_exit_loc = tour_locs[-1] if tour_locs else prev_exit_loc

    # Check for any missed venues
    visited_names = set(v.name for v in ordered_venues)
    for v in venues:
        if v.name not in visited_names:
            print(f"  WARNING: Missed venue: {v.name} (region: {v.tour_region})")
            ordered_venues.append(v)

    return ordered_venues


# ─── Day Splitting ───────────────────────────────────────────────────────────

def split_into_days(ordered_venues: list[Venue], dist: np.ndarray) -> list[dict]:
    """Split continuous venue list into days respecting max daily miles."""
    if not ordered_venues:
        return []

    days = []
    day_venues = [ordered_venues[0]]
    day_dist = 0.0

    for i in range(1, len(ordered_venues)):
        seg = dist[ordered_venues[i-1].loc_id][ordered_venues[i].loc_id]

        # If this venue is at the same location as previous, add it for free
        if ordered_venues[i].loc_id == ordered_venues[i-1].loc_id:
            day_venues.append(ordered_venues[i])
            continue

        # Would adding this exceed the limit?
        if day_dist + seg > MAX_DAILY_MILES and len(day_venues) > 1:
            # Close current day
            days.append({
                "venues": day_venues,
                "distance": day_dist,
            })
            # New day starts at the last venue of previous day
            day_venues = [day_venues[-1]]
            day_dist = 0.0

        day_venues.append(ordered_venues[i])
        day_dist += seg

    # Close final day
    if day_venues:
        days.append({"venues": day_venues, "distance": day_dist})

    return days


# ─── Output ──────────────────────────────────────────────────────────────────

def write_route_json(days: list[dict], path: str):
    """Write route in the same format as route_plan_v2.json."""
    mainland_days = []

    for day_num, day in enumerate(days, 1):
        stops = []
        seen = set()
        for v in day["venues"]:
            if v.name in seen:
                continue
            seen.add(v.name)
            stops.append({
                "name": v.name,
                "category": v.category,
                "postcode": v.postcode,
                "address": v.address,
                "lat": v.lat,
                "lon": v.lon,
            })

        if not stops:
            continue

        mainland_days.append({
            "day": day_num,
            "distance_miles": round(day["distance"], 1),
            "num_stops": len(stops),
            "start": stops[0]["name"],
            "end": stops[-1]["name"],
            "stops": stops,
        })

    output = {"mainland_days": mainland_days, "ni_days": []}

    with open(path, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    return output


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UK Sports Club Bike Tour - Route Optimizer v3")
    print("=" * 60)

    # 1. Load venues
    print("\n1. Loading venues...")
    venues = load_venues("geocoded_venues_v2.json")
    print(f"   Loaded {len(venues)} venues")

    # Show region distribution
    region_counts = {}
    for v in venues:
        region_counts[v.tour_region] = region_counts.get(v.tour_region, 0) + 1
    for r in REGION_ORDER:
        print(f"   {r}: {region_counts.get(r, 0)} venues")
    for r, c in region_counts.items():
        if r not in REGION_ORDER:
            print(f"   UNASSIGNED - {r}: {c} venues")

    # 2. Build unique locations
    print("\n2. Building unique locations...")
    locations, loc_to_venues = build_unique_locations(venues)
    print(f"   {len(locations)} unique physical locations")

    # 3. Build distance matrix
    print("\n3. Building distance matrix...")
    dist = build_distance_matrix(locations)

    # 4. Build route
    print("\n4. Building optimized route...")
    ordered = build_route(venues, locations, loc_to_venues, dist)
    print(f"   Total venues in route: {len(ordered)}")
    unique_names = set(v.name for v in ordered)
    print(f"   Unique venue names: {len(unique_names)}")

    # 5. Split into days
    print("\n5. Splitting into days...")
    days = split_into_days(ordered, dist)

    # 6. Summary
    total_miles = sum(d["distance"] for d in days)
    total_venues = len(set(v.name for d in days for v in d["venues"]))
    print(f"\n{'=' * 60}")
    print(f"ROUTE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total days: {len(days)}")
    print(f"Total miles: {total_miles:.0f}")
    print(f"Total unique venues: {total_venues}")
    print(f"Average miles/day: {total_miles/len(days):.1f}")
    print()

    long_days = []
    for i, day in enumerate(days):
        d_num = i + 1
        vs = day["venues"]
        # Unique venue names in this day
        names = []
        seen = set()
        for v in vs:
            if v.name not in seen:
                names.append(v.name)
                seen.add(v.name)

        flag = " ⚠️  OVER 90mi" if day["distance"] > 90 else ""
        print(f"  Day {d_num:2d}: {names[0]:30s} → {names[-1]:30s}  "
              f"{day['distance']:6.1f} mi  {len(names):2d} stops{flag}")

        if day["distance"] > 90:
            long_days.append((d_num, day["distance"], names[0], names[-1]))

    if long_days:
        print(f"\n⚠️  {len(long_days)} days exceed 90 miles:")
        for d_num, miles, start, end in long_days:
            print(f"   Day {d_num}: {miles:.1f} mi ({start} → {end})")

    # 7. Write output
    print(f"\n6. Writing route_plan_v3.json...")
    output = write_route_json(days, "route_plan_v3.json")
    print(f"   Done! {len(output['mainland_days'])} days written.")

    return output


if __name__ == "__main__":
    main()
