# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Azizghandos/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Azizghandos/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast
each GTA VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z,
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per trophy.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic).
It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Model (see the docstring in `main.py` for the full reasoning)

An Exophase rate is earners / tracked players, and both move at different
speeds. Launch day sold ~24.6M copies (Sensor Tower estimate, ~75% on PS5)
while Exophase tracked only 638 players on launch night, so the tracked
population is still growing fast and every new tracked player dilutes a
trophy's rate. Exophase also refreshes on its own schedule — the page sat
unchanged from 2026-11-18T23:48Z to the 04:00Z read, then jumped to 3,868
tracked at 04:30Z with almost every rate lower.

The forecaster therefore:

1. Decomposes each rate into earners E(t) and tracked players N(t), fits
   each in log space against log time since release, and takes the median
   as E(target) / N(target).
2. Fits the logit of the rate against log time as a second view and
   averages the two medians in logit space.
3. Damps the extrapolation and caps the median's move from the last read.
4. Uses a wide, honest spread: a random-walk term growing with the
   horizon, a refresh-shock floor, fit residuals, and slope uncertainty.
5. If a case's history is one perfectly flat, unrefreshed page (launch
   night), applies a stale-page dilution prior to the median and widens
   the spread — that branch cannot fire on a multi-day final case.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=660zm28c&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
