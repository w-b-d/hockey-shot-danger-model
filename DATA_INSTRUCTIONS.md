# Getting MoneyPuck Shot Data

This project is designed for MoneyPuck's public NHL shot-level CSV files. The files are free to download from [moneypuck.com/data.htm](https://moneypuck.com/data.htm), require no API key, and include saved shots, missed shots, goals, adjusted coordinates, game context, and MoneyPuck model columns such as `xGoal`.

MoneyPuck asks users to credit MoneyPuck.com when using the data and to avoid unapproved scraping. This project only uses the published downloadable CSV files.

## Recommended Start: One Completed Season

As of May 5, 2026, the 2024-25 shot file is the latest completed-season file listed in the public shot-data section. Download it and save it as `data/shots.csv`:

```bash
cd data
curl -LO https://peter-tanner.com/moneypuck/downloads/shots_2024.zip
unzip shots_2024.zip
mv shots_2024.csv shots.csv
cd ..
```

If MoneyPuck changes a URL, go to the data page, download the desired file under **Shot Data**, unzip it, and rename the CSV to `data/shots.csv`.

## Current Season Data

MoneyPuck's data page says 2025-26 shot data is available and updated nightly. Because current-season links can change, use the exact current-season shot-data link shown on MoneyPuck's page rather than hard-coding it in a resume project.

## Multiple Seasons

MoneyPuck also lists bundled historical files, including:

- all past seasons
- recent seasons
- individual season files

Multi-season files are useful for model training, but they are larger and will make the first Streamlit load slower. A single season is enough to demonstrate the full pipeline.

## Expected Columns

The loader auto-detects common MoneyPuck names and a few common alternatives. The most important fields are:

| Purpose | Common MoneyPuck column |
|---|---|
| Shot x-coordinate | `xCordAdjusted` or `arenaAdjustedXCord` |
| Shot y-coordinate | `yCordAdjusted` or `arenaAdjustedYCord` |
| Fallback distance | `shotDistance` |
| Goal flag | `goal` |
| Shot type | `shotType` |
| Rebound / rush | `shotRebound`, `shotRush` |
| Empty net | `shotOnEmptyNet` |
| Strength state | `isHomeTeam`, `homeSkatersOnIce`, `awaySkatersOnIce` |
| Score state | `homeTeamGoals`, `awayTeamGoals` |
| Team filter | `teamCode` |
| Shooter | `shooterName` |
| Benchmark | `xGoal` |

Optional bonus columns used when present:

- `offWing`
- `speedFromLastEvent`
- `shotAnglePlusRebound`
- `playerPositionThatDidEvent`

## Synthetic Fallback

If you want to test the code before downloading anything:

```bash
python make_sample_data.py
```

This creates `data/sample_shots.csv`. It is useful for smoke tests and dashboard demos, but it is not NHL evidence and should not be used for portfolio claims.

## Troubleshooting

**"Missing shot geometry"**  
You may have downloaded a skater, goalie, team, or line file instead of a shot file. Download from the **Shot Data** section.

**"Missing goal indicator"**  
The project needs a binary `goal` column or an event column with labels such as `GOAL` and `SHOT`.

**Dashboard is slow**  
Start with one season instead of a multi-season file. Streamlit caching helps after the first load.

**Team filter is missing**  
The dashboard and Tampa Bay example need a team column such as `teamCode`. The model can still train without it, but team comparisons will be skipped.

