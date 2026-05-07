# Tampa Bay Lightning Public-Data Example

## Summary

In the 2025 MoneyPuck shot dataset, Tampa Bay generated slightly higher shot danger per shot than the rest of the league, about +0.003 to +0.004 xG per shot depending on whether using the project model or MoneyPuck's published `xGoal`. This is public-data analysis, not proprietary team insight or a full team evaluation.

## Verified 2025 Results

| Measure | Tampa Bay | Rest of league | Difference |
|---|---:|---:|---:|
| Project model predicted xG/shot | 0.0717 | 0.0680 | +0.0037 |
| MoneyPuck `xGoal`/shot | 0.0716 | 0.0684 | +0.0032 |

The agreement between the compact project model and MoneyPuck's published `xGoal` is useful: both point to a slightly better-than-league-average shot-danger profile for Tampa Bay in this public dataset. The effect size is modest, so the right interpretation is "slightly higher shot quality per shot," not a sweeping claim about overall team strength.

## What It Means For A Coach

This result suggests Tampa Bay's shot profile created marginally better looks on average than the rest-of-league baseline in the 2025 public shot data. The next coaching question is where that edge came from: more slot looks, more rebound chances, better shot selection by specific lines, power-play effects, or game-state effects.

## Interpretation Cautions

- This is based on public MoneyPuck shot data, not private tracking, internal video tags, or proprietary team data.
- The public `shotRush` flag is extremely sparse in this dataset, including only two Tampa Bay rush-flagged shots after cleaning, so it should not be treated as a complete transition-offense measure.
- A +0.003 to +0.004 xG/shot edge is meaningful over a large sample, but small enough that it should be paired with video review and opponent/context splits before drawing tactical conclusions.
- MoneyPuck's published `xGoal` outperformed the compact project model overall, so this project should be framed as an interpretable baseline and communication tool.

## Useful Follow-Up Questions

- Which Lightning skaters or lines drove the higher average shot danger?
- Did the edge come mostly at even strength or on special teams?
- Were the higher-danger looks created by rebounds, slot entries, net-front traffic, or specific shot types?
- Does the same pattern hold across multiple seasons, or is it specific to the 2025 file?

