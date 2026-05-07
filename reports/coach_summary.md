# Shot Danger Coach Summary

## Overview

This project estimates shot danger as the probability that a shot becomes a goal, using public MoneyPuck shot data and transparent geometry/context features. The goal is not to replace video or private tracking data; it is to give coaches and analysts a common language for discussing the quality of looks created and conceded.

## Three key findings

1. **Location drives most of the signal.** Shots from the slot and near the crease carry far more scoring value than perimeter volume. Distance is not linear: moving from 10 to 20 feet matters much more than moving from 45 to 55 feet.
2. **Rebounds are valuable, but quality still matters.** In the verified 2025 MoneyPuck file, rebound shots had a substantially higher raw goal rate than non-rebound shots, about 11.04% vs. 6.45%. Their modeled value still depends on location, angle change, and rebound quality.
3. **Shot type needs context.** Deflections showed the highest raw shot-type goal rate in the 2025 file, while tip-ins were more context-dependent after accounting for location and shot context. The coefficient table should be read as "all else equal," not as a raw ranking of shot types.

## Three tactical recommendations

1. **Defend the middle first.** The model reinforces a simple coaching principle: protect the slot, force shots wide, and make opponents finish possessions from lower-danger locations.
2. **Treat rebounds as second possessions.** Box-outs and recovery routes after the first save matter because rebound shots scored at a meaningfully higher raw rate in the 2025 data. The next layer of analysis should separate harmless loose pucks from lateral, high-angle-change rebounds.
3. **Use traffic with precision.** Point shots are most useful when they create screens, deflections, or rebounds. Deflections showed elevated raw value, but tip-ins and net-front plays should be evaluated with location, angle, and pre-shot context rather than by shot label alone.

## Limitations

- **No private tracking data.** The model does not see screens, passing lanes, defender pressure, goalie movement, or exact pre-shot puck movement.
- **Sparse rush flag.** The public `shotRush` flag was extremely sparse in the verified 2025 file, so rush-related results should be treated as a rough public-data proxy rather than a complete transition-offense measure.
- **No goalie or shooter talent adjustment.** The model estimates shot quality, not finishing talent or save difficulty for a specific goalie.
- **Public-data team examples are directional.** Tampa Bay and other team comparisons are useful for asking better video questions, not for making proprietary team evaluations.

## Next steps with private tracking data

- Add pre-shot puck movement: east-west passes, time since pass, and slot-line movement.
- Add goalie movement and state: lateral distance traveled, recovery status, and screen/traffic context.
- Add defender geometry: nearest defender distance, sticks in lane, and net-front body position.
- Split rebounds by quality: original shot location, rebound distance, angle change, and time between shots.

