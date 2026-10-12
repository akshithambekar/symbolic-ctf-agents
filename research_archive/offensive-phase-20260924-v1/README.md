# Phase 3: offensive recovery

This is a fresh exploratory phase. The completed Phase 2 suite is an immutable
input and is never modified. First, a scripted oracle tests physical feasibility
and identifies the failing offensive stage. Then three low-cost variants use two
seeds at 30,720 transitions. A variant advances only after at least 10% captures
over 40 held-out episodes at the easier stage. Failed variants stop, preventing
an expensive matrix with no evidence of learnability.

The mission specification retains `G(in_bounds)` and ordered pickup-before-capture,
but removes the absorbing no-tag and global fixed-defender clauses. Tagging remains
a measured outcome and task penalty. `soft_guidance` increases pickup/capture signal,
reduces tag and time penalties, removes absorbing-failure penalties, and keeps the
temporal-state input. This phase is exploratory; it cannot support superiority claims.
