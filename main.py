"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) of the rate in percent.

What the data actually is
--------------------------
Exophase's rate for a trophy is earners / tracked players, where "tracked
players" is the small, self-selected set of players who have synced their
trophies with Exophase.  Both the numerator and the denominator move, and
they move at very different speeds:

* Launch day sold ~24.6M copies (Sensor Tower estimate; ~75% on PS5, so
  ~18M PS5 players), while Exophase tracked only 638 players on launch
  night.  The tracked population is therefore still growing fast, and every
  new tracked player who has not yet earned a trophy dilutes that trophy's
  rate.  A rate can fall even while more players are earning the trophy.
* Exophase refreshes its counts on its own schedule, not continuously: the
  page sat unchanged (638 tracked) from 2026-11-18T23:48Z through the
  04:00Z read on launch night, then refreshed at 04:30Z to 3,868 tracked
  with almost every rate lower.  Rates move in refresh jumps, not smoothly.

So a forecast that just persists the last rate (the house baseline) is
wrong in two ways: it misses the direction of travel, and its spread is far
too narrow to cover a refresh jump.  This program instead:

1. Decomposes each trophy's rate into earners E(t) and tracked players N(t)
   and fits each separately in log space against log time since release.
   The median rate is E(target) / N(target): dilution is modelled, not
   assumed away.
2. Fits the logit of the rate directly against log time as a second,
   independent view, and averages the two medians in logit space.
3. Damps the extrapolation (a trend fitted over 2 days should not be
   projected 4.5 days forward at full strength) and caps how far the
   median may move from the last read.
4. Uses an honest, wide spread: a random-walk term growing with the
   horizon, a refresh-shock floor (Exophase jumps are ~1 logit unit), the
   fit's residual variance, and the slope's uncertainty.  On the logit
   scale a miss from 1% to 2% costs about what a miss from 33% to 50%
   does, so rare trophies get the same care as common ones.
5. Handles the stale-page case specially: when a case's whole history is
   one unchanged read (launch night, before the first refresh), there is
   no trend to fit, but the page is known to be stale - the tracked
   population is about to be diluted by the launch-day wave.  The median
   is then diluted by a prior factor and the spread widened.  This branch
   only fires on a perfectly flat history, which a multi-day final case
   cannot have.

No network, no randomness, standard library only.  Deterministic: the same
input bytes always produce the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release: 2026-11-19T00:00:00Z (midnight UTC start of launch day).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread of a rate's logit, per sqrt(day) -- the house baseline's.
SIGMA_RW = 0.5
# Refresh-shock floor: Exophase's own refreshes move logits by ~1 unit.
SHOCK_FLOOR = 0.55
# Minimum / maximum logit spread of any forecast.
SPREAD_MIN, SPREAD_MAX = 0.75, 2.2
# How far the median logit may move from the last read, in logit units.
MEDIAN_MOVE_CAP = 3.0
# Recency half-life for the trend fits, in hours.
HALF_LIFE_H = 18.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b, resid_var)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return (a, b, resid)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    u_target = math.log1p(hours_since_release(target))

    # Deduplicate reads by timestamp, keeping the last of each.
    seen = {}
    for read in history:
        seen[read["at"]] = read
    reads = [seen[k] for k in sorted(seen)]
    last = reads[-1]
    t_last = hours_since_release(parse_time(last["at"]))
    u_last = math.log1p(t_last)
    n_last = float(last["players_tracked"]) if last["players_tracked"] else 0.0

    # Is the whole history one unchanged page? (launch night, pre-refresh)
    flat = all(r["rates"] == last["rates"] and r["players_tracked"] == last["players_tracked"] for r in reads)

    # --- Fit the tracked-population curve log N ~ a + b*u (recency weighted).
    n_xs, n_ys, n_ws = [], [], []
    for r in reads:
        n = float(r["players_tracked"]) if r["players_tracked"] else 0.0
        if n <= 0:
            continue
        t = hours_since_release(parse_time(r["at"]))
        n_xs.append(math.log1p(t))
        n_ys.append(math.log(n))
        n_ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
    n_fit = wls(n_xs, n_ys, n_ws) if len(n_xs) >= 2 else None

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Per-trophy series.
        xs, lgs, es, ws = [], [], [], []
        for r in reads:
            rate = r["rates"][i]
            if rate is None:
                continue
            t = hours_since_release(parse_time(r["at"]))
            n = float(r["players_tracked"]) if r["players_tracked"] else 0.0
            xs.append(math.log1p(t))
            lgs.append(logit(rate))
            es.append(max(rate / 100.0 * n, 0.5))
            ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        last_lg = logit(last_rate)

        if len(xs) < 2 or flat:
            median_lg = last_lg
            resid = 0.0
            slope_var = 0.0
        else:
            # View 1: direct logit trend, damped at the target.
            fit = wls(xs, lgs, ws)
            a1, b1, resid = fit
            # Damp the slope the further past the data we extrapolate.
            du = max(0.0, u_target - u_last)
            damp = 1.0 / (1.0 + 0.35 * du)
            lg_a = a1 + b1 * u_target if fit else last_lg
            lg_a = last_lg + (lg_a - last_lg) * damp
            # View 2: earners / tracked decomposition.
            lg_b = None
            if n_fit is not None and len(xs) >= 2:
                e_fit = wls(xs, [math.log(e) for e in es], ws)
                if e_fit is not None:
                    ea, eb, _ = e_fit
                    # Earners grow, but saturate: damp their slope too.
                    log_e_t = ea + eb * u_target * 1.0
                    log_e_t = math.log(es[-1]) + (log_e_t - math.log(es[-1])) * damp
                    log_n_t = n_fit[0] + n_fit[1] * u_target
                    log_n_t = math.log(max(n_last, 1.0)) + (log_n_t - math.log(max(n_last, 1.0))) * damp
                    rate_b = 100.0 * math.exp(log_e_t - log_n_t)
                    lg_b = logit(rate_b)
            if lg_b is None:
                median_lg = lg_a
            else:
                median_lg = 0.5 * (lg_a + lg_b)
            # Slope uncertainty grows with distance past the data.
            slope_var = (0.25 * du) ** 2

        # Cap how far the median may move from the last read.
        median_lg = last_lg + max(-MEDIAN_MOVE_CAP, min(MEDIAN_MOVE_CAP, median_lg - last_lg))

        # Stale-page prior: flat history means Exophase has not refreshed
        # since before launch; the launch-day wave is about to dilute the
        # tracked cohort.  Dilute the median (except the first trophy,
        # which every tracked player holds) and widen the spread.
        spread_extra = 0.0
        if flat:
            if last_rate >= 99.9:
                median_lg = logit(99.0)
            elif trophy["tier"] in ("platinum", "gold") and trophy["id"] in (1, 14):
                median_lg = last_lg + math.log(0.12)
            else:
                median_lg = last_lg + math.log(0.30)
            spread_extra = 1.15 ** 2

        # First-trophy special case: a rate of 100% is earned on starting
        # the game; new joiners hold it too, so keep it near the top but
        # allow some dilution.
        if last_rate >= 99.9 and not flat:
            median_lg = max(median_lg, logit(97.0))

        # Residual variance only exists when a trend was actually fitted.
        resid_term = 0.5 * resid if not flat else 0.0
        spread = math.sqrt(
            (SIGMA_RW * math.sqrt(horizon_days)) ** 2
            + SHOCK_FLOOR ** 2
            + resid_term
            + slope_var
            + spread_extra
        )
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))

        qs = [round4(clamp(inv_logit(median_lg + z * spread))) for z in Z]
        # Keep quantiles non-decreasing (rounding can tie them, never invert).
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
