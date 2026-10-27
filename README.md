# Cajetan

Cajetan rates NFL parlays. You give it the legs of a bet, such as `Josh Allen o 34.5 rush yds` or `BUF -3.5`. For every leg it returns a **confidence rating out of 100** (the estimated chance that leg wins), the **reasons** behind the number, and a combined rating for the **whole parlay**. The parlay rating accounts for legs from the same game moving together.

Ratings come from free nflverse data (play-by-play, FTN charting, NFL participation, weekly stats, snap counts, injury reports, Next Gen Stats, schedules and betting lines) plus free weather forecasts. They combine four things:

1. A matchup engine that compares a player's production in his team's schemes and personnel against what the opposing defense allows in those same situations.
2. Situational adjustments: red zone, late game, running quarterbacks, the injury report (his own status and teammates ruled out) and weather.
3. A purely historical gradient-boosting model trained on every player-game since 2016.
4. Weights and a final probability calibration fitted on a leak-free backtest and on your graded past ratings.

```
$ cajetan rate "James Cook 80+ rush yds" "Dalton Kincaid under 3.5 rec" "BUF ML -180"

 # | Leg                                 | Matchup                                | Proj        | Confidence | Data | Verdict
 1 | James Cook over 79.5 rushing yards  | James Cook (RB, BUF vs BAL (wk 8))     | 92.5        | 56%        | A    | lean
 2 | Dalton Kincaid under 3.5 receptions | Dalton Kincaid (TE, BUF vs BAL (wk 8)) | 3.5         | 55%        | A    | lean
 3 | BUF moneyline                       | BUF moneyline (BUF vs BAL (wk 8))      | +2.7 margin | 58%        | A    | lean

Leg 1: James Cook over 79.5 rushing yards - 56%
  Projects 92.5 rushing yards on average (median 86.6) vs a line of 79.5 -> 56% to go over.
  Rated high: median outcome 86.6 is above the 79.5 line; partly offset by Late-game defense, Defense vs position, Volume allowed.
  [-] Late-game defense (-3.4%): BAL allows 2.94 yds/carry in the 4th quarter/OT (league 4.02); 39% of Cook's volume
      comes late; game script points to more late carries
  [-] Defense vs position (-3.4%): BAL allows 3.97 yds/carry to RBs (league 4.20; 8th-stingiest of 32, 125 plays this season)
  [-] Volume allowed (-2.7%): BAL concedes 20.3 RB carries/game (league 21.9; 7th-fewest of 32)
  [-] Game script (-1.5%): BUF is favored by 3 (total 53.5); spread/total model expects 26.7 team carries vs 27.1 ...
  [+] Historical ML model (+1.3%): gradient-boosted model on 9,489 player-games (2022-2026) projects 99.10 ...
   .  Baseline volume: 19.8 carries/game (recency-weighted over 25 games, 6 this season)
   .  Baseline efficiency: 5.26 yds/carry on 117 carries this season; shrunk to 5.07 toward the RB average 4.20
  Data reliability A (96/100)
...
Overall parlay
  Parlay hit probability: 21.1%   (fair odds +373)
  If legs were independent: 17.8%   Average leg strength: 56%
```

> **Note:** Cajetan is a statistics tool, not financial advice. Sportsbooks build a margin into every price. Even well-calibrated probabilities lose money when the payout is worse than fair, and parlays multiply that margin. Bet responsibly.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Writing legs](#writing-legs)
- [Reading the output](#reading-the-output)
- [How a rating is built](#how-a-rating-is-built)
- [Data sources](#data-sources)
- [Validation](#validation)
- [Limitations](#limitations)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Development and tests](#development-and-tests)

---

## Installation

You need **Python 3.11 or newer**. It's developed on 3.13.

```powershell
cd Cajetan
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `cajetan` command. No API keys are needed, because all data comes from the public nflverse releases on GitHub.

## Quick start

```powershell
cajetan update          # first run downloads 2016-present (~10 min); later runs refresh just the current season
cajetan train --tune    # fits the ML models (searching their settings) + calibration (~20 min)
cajetan backtest        # rates last season and this one leak-free (~15 min on a multi-core PC)
cajetan fit             # fits blend weights, outcome spreads and the probability calibration from the backtest
cajetan rate "Bijan Robinson o 95.5 rush yds" "Jahmyr Gibbs anytime td +120" "GB/DET o 52.5"
```

Suggested weekly routine during the season:
1. Run `cajetan update` on Tuesday or Wednesday, after nflverse publishes the previous week's play-by-play. Run it again on Friday or Saturday to pick up the final injury report.
2. Run `cajetan train` (add `--tune` once a month or so).
3. Rate your parlays. Every rating is logged.
4. After the games, run `cajetan grade` to see how the logged ratings did at your real lines.
5. Every few weeks, run `cajetan backtest` and `cajetan fit` to refit the weights and calibration with the new weeks.

## Commands

| Command | What it does |
|---|---|
| `cajetan update [--full] [--seasons 2025,2026]` | Downloads nflverse data into `data/cajetan.duckdb`. The first run loads every season from 2016 on (play-by-play from 2022). After that only the current season is refreshed, unless you pass `--full` or `--seasons`. |
| `cajetan train [--tune]` | Builds the calibration (outcome distributions, dispersion, same-game correlations, weather, injury designations, vacated volume) and trains one gradient-boosting model per market. `--tune` also searches the model settings across held-out seasons. Results are saved to `data/models/`. |
| `cajetan backtest [--seasons 2025,2026] [--weeks 3-18]` | Rates past games using only data from before each one, with ML models retrained on earlier seasons, and scores the ratings per market. Saves `data/backtest.parquet`. |
| `cajetan fit` | Fits the ML blend weight per market, the market blend and outcome spreads for team bets, the game-log weighting and the final probability calibration from the backtest (and from graded logged ratings). Saves `data/models/tuning.json` and `prob_calibration.json`. |
| `cajetan grade` | Settles logged ratings whose games are final and saves them permanently in the `prediction_grades` table (won, lost, push, or void if the player didn't play). Then reports hit rate, Brier score, a calibration table and flat-bet profit for legs with odds, over the whole saved history. Rerunning it only adds new results and corrects any changed by stat corrections. It needs write access, so stop `cajetan web` first. |
| `cajetan rate LEG [LEG ...]` | Rates each leg and the overall parlay. |
| `cajetan rate -f parlay.txt` | Reads legs from a file, one per line. Lines starting with `#` are ignored. |
| `cajetan rate ... --odds +650` | Adds the book's parlay price, so Cajetan also reports edge and expected value. |
| `cajetan rate ... --week 5` | Rates a specific week of the current season. Only data from before that week is used, so you can check past results honestly. |
| `cajetan rate ... --json` | Prints machine-readable output, including every factor. |
| `cajetan rate ... --reasons 8` | Shows more reasons per leg (default 5). |
| `cajetan player "Ja'Marr Chase"` | Shows the player's per-game averages and scheme splits: share of volume and efficiency by run direction, formation, personnel, box count, play action, motion, coverage and blitz. |
| `cajetan defense Ravens` | Shows what a defense allows to each position, by scheme and personnel bucket, in the red zone and late in games. |
| `cajetan status` | Shows the data loaded per season, how each model scored on its validation seasons, and the fitted settings in effect. |
| `cajetan visits [--limit 30] [--ip 1.2.3.4] [--by-ip]` | Shows who connected to the web UI: time, IP, country, and the legs, odds and week they entered. `--by-ip` gives one row per visitor with first/last seen and counts. Stored in `data/access_log.sqlite`. Behind a Cloudflare Tunnel the visitor's real IP is read from Cloudflare's `CF-Connecting-IP` header (trusted only on connections from this machine); direct local visits show as `(local)`. |
| `cajetan web [--port 8765] [--no-open]` | Starts a local web UI at `http://127.0.0.1:8765/` and opens it in a browser. Enter legs (one per line), optional parlay odds and week, and see the same ratings, reasons and parlay summary as `rate`. |

## Writing legs

Each leg is one quoted string. Capitalisation, extra spaces and most abbreviations don't matter. You can put American odds at the end (`-115`, `+120`, `(-110)`), and they're used for the edge and expected-value math.

**Player props**

| You type | Parsed as |
|---|---|
| `Josh Allen o 34.5 rush yds` / `over 34.5 rushing yards` | rushing yards over 34.5 |
| `Ja'Marr Chase u6.5 receptions` | receptions under 6.5 |
| `Jared Goff 250+ pass yds` | passing yards over 249.5 |
| `Jalen Hurts 2+ passing tds` | passing TDs over 1.5 |
| `Bijan Robinson o 119.5 rush + rec yds` | rushing + receiving yards |
| `Saquon Barkley over 21.5 carries` | rush attempts |
| `Patrick Mahomes over 24.5 completions` / `o 36.5 pass attempts` / `u 0.5 ints` | QB volume and turnovers |
| `Jahmyr Gibbs anytime td` / `ATD` / `to score a touchdown` | anytime touchdown |

**Team markets.** Use a team abbreviation, city or nickname.

| You type | Parsed as |
|---|---|
| `BUF ML`, `Bills moneyline -150`, `Kansas City Chiefs` | moneyline |
| `DET -3.5`, `Lions +3 -110`, `BUF pk` | spread |
| `BUF/MIA o 47.5`, `Packers @ Lions under 52.5 total` | game total |
| `BUF team total over 26.5` | team total |

Player names are matched loosely. "Kenneth Walker", "K Walker" and small typos all work. If two active players share a name, the one at a position that fits the market wins. Each leg is rated against that team's **next unplayed game** unless you pass `--week`.

## Reading the output

- **Confidence** is the estimated probability, as a percentage, that the leg wins. Pushes on whole-number lines are refunded, so they're removed from the calculation, and the push chance is listed separately.
- **Verdict** groups the confidence into bands:
  - `strong` is 65% or higher
  - `lean` is 55–64%
  - `coin flip` is 45–54%
  - `fade` is 35–44%
  - `avoid` is under 35%
- **Data grade (A–D)** describes how much evidence the rating stands on, separately from the confidence. It considers the player's games this season, the opponent's games, whether the scheme splits use this season's charting, and whether the ML model and calibration are trained. A 70% with grade D deserves more skepticism than a 70% with grade A.
- **Proj** is the projected average. For yardage, the summary also gives the **median**, because yardage is right-skewed: a few big games pull the average up, and sportsbooks set lines near the median.
- **Reasons:**
  - `[+]` helps your side of the bet.
  - `[-]` hurts it.
  - ` . ` is context (baselines, red-zone and late-game splits).

  Each reason shows its effect on the projection (`+4.1%`, or `+1.5 pts` for team markets) and the evidence behind it.
- **"Rated high/low"** is a one-line explanation: where the median lands against the line, then the biggest adjustments for and against.
- **Overall parlay:**
  - **Parlay hit probability** is the chance every rated leg wins, including the same-game correlation adjustments.
  - **If legs were independent** shows the plain product for comparison.
  - **Average leg strength** is the geometric mean of the leg probabilities.
  - If you supply odds, it also shows the implied probability, the edge, and the expected value per unit staked.
  - The weakest leg and any significant correlations are listed.

---

## How a rating is built

### Data window

Every rating uses only games played **before** the game being rated. Where a stat is pooled, the current season counts fully and the previous season counts at a lower weight (currently 0.6, fitted by `cajetan fit`; 0.25 by default), so early-season ratings still have a sample. Within a player's game log, recent games count more: a game's weight halves every few games (currently 3, fitted; 4 by default).

### Player props: volume × efficiency

A projection is broken down into **volume** (attempts, carries or targets) and **efficiency** (yards per attempt, carry or target, catch rate, completion rate, TD rate or INT rate):

```
projection = volume × game-script and volume-allowed adjustments
           × efficiency × defense, scheme and situational adjustments
```

**1. Baseline**
- Volume is the recency-weighted per-game average. Kneel-downs are removed from QB carries.
- Efficiency is the player's per-play rate, shrunk toward the average for his position. Shrinkage is an empirical-Bayes technique: a player with 30 carries is pulled harder toward average than one with 300.

**2. Volume adjustments**
- **Game script:** a regression on every game since 2022 estimates how a team's pass attempts and carries change with the betting spread and total. For example, favored teams run more and underdogs throw more. Cajetan compares this game's spread and total against the player's usual game environment.
- **Volume allowed:** how many carries or targets the opponent gives up per game to this position, compared with the league. This counts at half weight, because some of it is game script.
- **Teammates out:** if teammates ruled Out or Doubtful on this week's injury report played the previous game, their season share of targets or carries is freed up. An even redistribution would give this player `1 / (1 − vacated share)` times his usual volume; `cajetan train` measures how much of that players have historically absorbed (about 55% for targets and 52% for carries) and applies that much. This factor isn't capped with the others, because an injured starter can legitimately double a backup's role.

**3. Defense vs position.** This is the opponent's rate allowed to the player's position (for example, yards per carry allowed to RBs), shrunk and ranked 1–32.

**4. Scheme and personnel matchups.** This is the core of the matchup engine. Each play is tagged with:

| Dimension | Buckets | Chosen by | Source |
|---|---|---|---|
| Run direction | outside (end) / off-tackle / inside (guard, middle) | offense | play-by-play run gap |
| Formation | shotgun / under center / pistol | offense | FTN, participation |
| Personnel | 11, 12, 21, 13, 22, ... | offense | NFL participation |
| Play action, pre-snap motion | yes / no | offense | FTN charting |
| QB run type | designed / scramble / sneak | offense | play-by-play, FTN |
| Box count | light (≤6) / 7 / stacked (8+) | defense | FTN, participation |
| Coverage | man / zone | defense | NFL participation |
| Blitz | blitz / no blitz | defense | FTN charting |

- **Offense-chosen dimensions:** Cajetan measures how much of the player's production comes from each bucket. Then it measures how the opponent defends that bucket, relative to how good it is overall.
  - Example: 60% of an RB's carries are outside runs, and this defense allows 5.9 YPC on outside runs against a league average of 4.6. That's well beyond its overall level, so the projection goes up.
  - Example: a TE whose targets mostly come in 12 personnel faces a defense that is weak in coverage against 12 personnel.
- **Defense-chosen dimensions:** Cajetan compares how often this defense shows a look with how often the player usually sees it, then weights by how he performs against that look.
  - Example: a defense blitzes on 41% of dropbacks, but the QB has typically seen blitzes on only 25%, and he averages 6.1 YPA against the blitz versus 7.4 overall.
- **Combining dimensions:** the dimensions overlap, since personnel and formation are related, so their effects are partially pooled at 60% and capped at ±18%.

**5. Situational defense**
- **Late game:** the opponent's 4th-quarter/OT efficiency allowed, relative to its overall level. This is weighted by how much of the player's volume comes late, and more when the game script points that way: carries when his team is favored, passes when it's the underdog.
- **Running QBs** (QB rushing props): how the opponent has done against QBs who average 25+ rushing yards per game, compared with **those QBs' own averages**. This adjusts for the opponents they faced. The output also states whether the QB in question is himself a running QB.
- **Red zone** (TD props): see below.

**6. Injury report and weather**
- **The player's own designation:** if he's listed Questionable or Doubtful, the projection is scaled by how players with that designation who then played actually produced against their season average, measured on every player-game since 2016. For example, Questionable pass catchers averaged about 13% fewer receiving yards. A player listed Out gets a warning, since sportsbooks usually void props on inactive players.
- **Weather:** recorded kickoff conditions for played games, and an [Open-Meteo](https://open-meteo.com/) forecast (free, no key) for upcoming outdoor games within about two weeks. `cajetan train` measures each condition from team games since 2016, as a team's stat that day against its season average before it. For example, wind of 15+ mph cuts passing yards by about 7%, and 40°F or colder cuts them by about 4%. Domes and closed roofs count as indoor games. If the forecast can't be fetched, the leg is rated without weather.
- If the week's injury report isn't loaded yet, the leg says so. Run `cajetan update` again once it's published (Wednesday to Friday).

**7. Anytime TD and passing TDs.**

```
expected TDs = carries × (red-zone share × TD rate inside the 20 + rest × TD rate outside)
             + targets × (same for targets)
```

Each rate is shrunk toward the positional average. The result is then adjusted by:
- the opponent's red-zone TD rate allowed to the position, capped at ±15%
- the scoring environment: market-implied team points against the team's average

The probability of scoring is `1 − e^(−expected TDs)` (Poisson).

**8. Historical ML model.**
- One `HistGradientBoostingRegressor` per market is trained on every player-game since 2016 (weekly stats go back further than play-by-play, which the matchup engine needs).
- **Features:**
  - season-to-date and last-3-game averages for 12 stats, plus the previous season's
  - the player's role: target share, air-yard share, carry share and snap share
  - season-to-date Next Gen Stats: separation, cushion, intended air yards, YAC over expected, rush yards over expected, stacked-box rate, completion % over expected, time to throw and aggressiveness
  - the week's injury report: his own designation and practice status, and the target and carry share vacated by teammates ruled out
  - what the opponent had allowed to the position so far that season relative to league
  - age and years in the league, home/away, spread, total, implied points, wind, temperature and indoor/outdoor
- Every feature uses only data from before kickoff.
- **Validation is rolling-origin:** for each of the last three complete seasons, the model is trained on everything before it and scored on it against a naive "season average so far" guess. `cajetan train --tune` uses the same folds to choose the model settings (learning rate, tree size, leaf size, regularisation) per market.
- The ML projection's share of the final projection is fitted per market by `cajetan fit`, from backtest log loss (see `cajetan status`). Before the first fit, it's 35% where the model beats the naive guess and 15% where it doesn't.

**9. From projection to probability**
- **Yardage props** use **empirical outcome distributions**. From every player-game since 2022, Cajetan computes `actual ÷ pre-game season average` and stores the distribution of that ratio by market, position and expected-volume tier. Your projection is scaled by it. This captures the real shape of outcomes: right skew, zero-catch games, and lower-volume players being relatively more volatile. If no calibration exists yet, a gamma distribution is used instead.
- **Count props** (receptions, attempts, completions, TDs, INTs) use a negative binomial distribution. Its overdispersion is estimated from historical data for each market and position.
- Half-point lines never push. Whole-number lines can push, and Cajetan uses a continuity correction to estimate that chance.

**10. Final calibration.** `cajetan fit` checks, per group of markets (yardage, counts, anytime TD, team bets), whether legs rated at a given probability actually hit that often in the backtest and in graded logged ratings. Where they don't, it fits a monotone correction curve (isotonic regression), shrunk toward "no change" when the sample is small. A group's curve is only switched on if it improves log loss in two-fold cross-validation, by season-week. When a leg is adjusted, a "Calibration" line in its reasons shows the raw and adjusted numbers.

### Team markets

- **Power ratings:** a weighted ridge regression on every score from this season and last. It fits an offense and a defense rating for each team plus home-field advantage, which gives opponent-adjusted projected points.
- **EPA per play:** the net offense − defense EPA/play difference × 62 plays gives a second margin estimate. The model margin is half power ratings, half EPA.
- **Weather** scales the model's projected points (not the market line, which already prices the forecast).
- **Market consensus:** the spread and total in the nflverse schedule are blended in. The betting market is a very strong predictor, and ignoring it would make the model overconfident. The blend share is fitted by `cajetan fit` (50% before the first fit).
- **Probabilities** use normal distributions. Their spreads (margin, total, team points) are fitted by `cajetan fit` as the spread of real results around the blended projections in the backtest. Before the first fit they're NFL-typical: margin SD 13.5, total SD 13, team points SD 9.5.
- **Context:** red-zone TD rate per trip (offense and defense) and 4th-quarter defensive EPA are reported alongside the result.

### Parlays and correlation

Legs from different games are treated as independent. Legs from the **same game** are joined with a Gaussian copula, using correlations that `cajetan train` estimates from history:

- **Same player:** completions ↔ passing yards ≈ 0.83, rushing yards ↔ anytime TD ≈ 0.40.
- **Teammates:** QB passing yards ↔ WR receiving yards ≈ 0.37, QB pass TDs ↔ WR anytime TD ≈ 0.38, RB ↔ RB carries are negative.
- **Opponents:** QB passing yards ↔ opposing QB passing yards ≈ 0.12.
- **Player stats vs game results:** RB rushing yards ↔ team margin ≈ 0.29, QB passing yards ↔ game total ≈ 0.35.

An under leg flips the sign. The joint probability is estimated with 200,000 simulated games. Positively correlated legs (QB over + his WR over) hit together more often than independent legs would. Negatively correlated legs make the parlay less likely, and the output names any correlation of 0.15 or more.

---

## Data sources

Data comes from [nflverse](https://github.com/nflverse) through the official [`nflreadpy`](https://github.com/nflverse/nflreadpy) package. It's free and needs no key.

| Dataset | Used for | Seasons |
|---|---|---|
| Play-by-play | every play-level split, EPA, red zone, late game, run direction | 2022– |
| FTN charting | play action, motion, RPO, screens, box count, blitzers, QB location | 2022– |
| NFL participation (NGS) | personnel groupings, formation, man/zone coverage, coverage shell, pressure | 2022–2025* |
| Weekly player stats | game logs, volume, target and air-yard share, ML training targets | 2016– |
| Schedules | scores, market spread and total, home/away, stadium, recorded kickoff weather | 2016– |
| Rosters | current team and position, age, years in the league, id mapping for snap counts | 2016– |
| Injury reports | game designation and practice status; teammates' vacated volume | 2016– |
| Snap counts | offensive snap share | 2016– |
| Next Gen Stats | separation, cushion, rush yards over expected, CPOE, time to throw, aggressiveness | 2016– |
| [Open-Meteo](https://open-meteo.com/) forecast | wind and temperature for upcoming outdoor games | live |

\* nflverse hasn't published participation data for the current (2026) season. Until it does, **personnel and coverage splits use 2025 data**. Each affected reason is marked `[uses 2025 data ...]`, its effect is halved, and the leg's data grade drops. All other current-season dimensions come from 2026 FTN charting and play-by-play. As soon as nflverse publishes 2026 participation data, `cajetan update` picks it up automatically.

## Validation

These checks were run on 2026-10-27, with data through week 7 of the 2026 season.

- **ML model, rolling-origin validation** (trained on seasons before each of 2023, 2024 and 2025, scored on it; mean absolute error, lower is better). Training from 2016 instead of 2022 gave the biggest gain; the new role, Next Gen Stats, injury and weather features added a smaller one; `--tune` a little more.

  | Market | Season-average guess | Previous model (2022+, old features) | Current model |
  |---|---|---|---|
  | Passing yards | 68.9 | 67.9 | 63.7 |
  | Pass attempts | 8.54 | 8.35 | 7.89 |
  | Rushing yards | 17.34 | 17.44 | 16.99 |
  | Rush attempts | 3.03 | 2.99 | 2.91 |
  | Receptions | 1.289 | 1.268 | 1.253 |
  | Receiving yards | 16.74 | 16.57 | 16.40 |
  | Rush + rec yards | 20.82 | 20.77 | 20.36 |
  | Anytime TD | 0.302 | 0.311 | 0.315 |

  The model now beats the naive guess everywhere except anytime TD by this measure (mean absolute error suits TD counts poorly; on backtest log loss the ML model does help anytime TD, and gets a 50% weight).
- **Backtest** (`cajetan backtest`): 13,592 legs from 2025 weeks 3–18 and 2026 weeks 3–7, rated leak-free. Every player market beat the "always predict the market's overall hit rate" baseline on Brier score and log loss. The team model **did not** beat the closing consensus line on spreads and totals, so `cajetan fit` set the market blend to 100%: team ratings now follow the market's projection, and the model's power ratings and EPA are shown as context. A team leg's confidence therefore moves only when your line differs from the consensus.
- **Out-of-sample check of the fitted weights:** fitted on 2025 backtest legs only and scored on the 2,835 player legs from 2026:

  | | Log loss | Brier |
  |---|---|---|
  | Always predict the hit rate | 0.6783 | 0.2426 |
  | Previous fixed weights (35% / 15% ML) | 0.6345 | 0.2221 |
  | Weights fitted on 2025 | **0.6263** | **0.2186** |

  Ratings line up with results across the range: legs rated 70.7% on average hit 72.3%, legs rated 49.4% hit 51.0%, and legs rated 25.9% hit 27.0%.
- **Probability calibration:** with the fitted weights, no group's correction curve improved cross-validated log loss, so calibration is currently off for all four groups. `cajetan fit` re-checks this every time, including graded logged ratings at real lines.
- **Game-log weighting:** the volume search preferred a 3-game half-life and a 0.6 prior-season weight over 4 and 0.25. The gain was small (next-game volume error 3.672 to 3.661 on 10,679 player-games).

Backtest props are rated at proxy lines (see Limitations), so these numbers say the ratings are well calibrated and informative, not that they beat sportsbook lines. Use `cajetan grade` on your logged bets to find that out.

## Limitations

- **Scheme labels are proxies.** No free dataset labels run concepts like outside zone, inside zone or power. Cajetan uses where the run went (end, tackle, guard or middle) as a proxy: "outside runs" include outside zone, stretch, toss and sweep. True concept labels would need a paid feed such as PFF or Sports Info Solutions. The `run_dir` column in `plays` is the only thing that would need to change.
- **Injury awareness is limited to the official report.** Designations, practice status and teammates ruled out are used, but news after the final report (a late scratch, a coach's comment about a reduced role) and players returning from injured reserve aren't. Run `cajetan update` after the Friday report.
- **Backtest lines are a proxy.** No free source has historical player-prop lines, so the backtest rates props at each player's season-to-date average, rounded to the half point below it. Books set sharper lines, so the backtest flatters the model somewhat. The prediction log (`cajetan grade`) measures accuracy at the real lines you enter, and graded legs feed the next `cajetan fit`.
- **Some calibration tables use every season.** The backtest retrains the ML models without the season being scored, but the outcome-distribution shapes, weather and injury effects come from all seasons. These are averages over thousands of games, so the leak is small.
- **Early-season samples are small.** Shrinkage and the down-weighted previous season keep weeks 1–3 sensible, but the data grade will be lower. Rookies and players who changed teams lean on positional averages.
- **Depth charts aren't used.** nflverse changed their format in 2025 (daily snapshots), and snap share plus the injury report capture most of the same information.
- **Weather forecasts** cover about two weeks ahead and use approximate stadium coordinates. Retractable roofs whose status isn't known yet get no weather adjustment.
- **Market lines** in the schedule are nflverse's consensus line, which may lag the line you're betting. If your number is different, that difference is exactly what the rating measures.
- Probabilities are estimates. A well-calibrated 60% still misses 4 times in 10.

## Configuration

Tunable constants live in `cajetan/config.py`. Those marked *fitted* are replaced by `cajetan fit`, which saves its values (and the evidence behind each) in `data/models/tuning.json`. Delete that file, or set the environment variable `CAJETAN_NO_TUNING=1`, to go back to the defaults.

| Setting | Default | Meaning |
|---|---|---|
| `FIRST_SEASON` | 2022 | Oldest season of play-by-play loaded. FTN charting starts in 2022. |
| `HISTORY_FIRST_SEASON` | 2016 | Oldest season of weekly stats, injuries, snaps and Next Gen Stats (for the ML model) |
| `PRIOR_SEASON_WEIGHT` *(fitted)* | 0.25 | Weight of last season's plays and games relative to this season's |
| `RECENCY_HALF_LIFE` *(fitted)* | 4.0 | Games until a game's weight halves in a player's log |
| `ML_BLEND_BY_MARKET` *(fitted)* | none | ML share of a player projection, per market |
| `ML_BLEND_WEIGHT` / `ML_BLEND_WEIGHT_WEAK` | 0.35 / 0.15 | ML share before the first fit (beat the naive guess in validation / didn't) |
| `MARKET_BLEND_WEIGHT` *(fitted)* | 0.5 | Share of the market spread or total in team projections |
| `MARGIN_SD`, `TOTAL_SD`, `TEAM_POINTS_SD` *(fitted)* | 13.5, 13, 9.5 | Spread of outcomes for team markets |
| `PARLAY_SIMULATIONS` | 200,000 | Monte Carlo draws for correlated parlays |

Per-market settings, such as which scheme dimensions apply and how much efficiency is shrunk, live in `cajetan/analysis/markets.py`. Set the `CAJETAN_HOME` environment variable to keep the `data/` folder somewhere else.

## Project layout

```
cajetan/
  cli.py              Typer commands (update, train, rate, player, defense, status)
  config.py           paths and tunable constants
  db.py               DuckDB connection + the as-of filter every query uses
  ingest.py           nflreadpy download -> DuckDB; builds derived `plays`, `players`, `team_games`,
                      `player_snaps`, `injury_status`, `ngs_week`, `player_shares`, `vacancy` tables
  parsing.py          free-text leg parser and odds helpers
  resolve.py          player-name matching and next-game lookup
  teams.py            team abbreviations and nicknames
  calibration.py      outcome distributions, dispersion and correlation estimates (cajetan train)
  engine.py           parse -> evaluate -> calibrate -> combine (and log)
  report.py           Rich terminal output and JSON
  profiles.py         `cajetan player` / `cajetan defense`
  web.py              `cajetan web` local server; page in static/index.html
  access_log.py       web visitor log (IP, time, legs) in data/access_log.sqlite (`cajetan visits`)
  weather.py          recorded conditions and Open-Meteo forecasts
  backtest.py         leak-free rating of past games (`cajetan backtest`)
  tuning.py           fits weights, spreads and game-log weighting (`cajetan fit`)
  probcal.py          final isotonic probability calibration
  predlog.py          prediction log and grading (`cajetan grade`)
  analysis/
    markets.py        market definitions (volume, efficiency, scheme dimensions, distribution)
    matchup.py        play-level splits, shrinkage, recency-weighted game logs
    environment.py    game-script regression, team scoring
    player_eval.py    player prop evaluator
    team_eval.py      moneyline / spread / totals evaluator
    ml.py             leak-free features + gradient boosting
    distributions.py  empirical, gamma, negative-binomial and normal outcome probabilities
    parlay.py         same-game correlation + Monte Carlo joint probability
    explain.py        Factor / LegResult / ParlayResult containers
tests/                parser, math, parlay, grading, calibration and end-to-end tests
data/                 (git-ignored) cajetan.duckdb, trained models, backtest.parquet, predictions.jsonl
```

The database is ordinary DuckDB. You can explore it directly with `duckdb data/cajetan.duckdb`, or from Python with `duckdb.connect("data/cajetan.duckdb")`. Key tables:

| Table | Contents |
|---|---|
| `plays` | One row per scrimmage play, with every scheme tag |
| `player_week` | Weekly player stats |
| `team_games` | Two rows per game: score, market margin and total, team pass attempts and carries |
| `schedules` | Schedule and betting lines |
| `players` | Latest name, position and team per player |
| `prediction_grades` | Every graded logged rating: the leg, line, odds, probability, actual result and when it was graded |

## Development and tests

```powershell
pytest            # parser, distribution, parlay-math unit tests + end-to-end tests (skipped if no data yet)
```

To add a new player market:
1. Add a `Market` entry in `analysis/markets.py`.
2. Add a parser pattern in `parsing.py`.
3. If it needs ML or correlation support, add its stat to `ml.STATS` and to `calibration.CORR_MARKETS`.
