-- Singular test: a day's minimum temperature must never exceed its maximum.
-- This cross-column rule cannot be stated by a single-column generic test, so it
-- lives here as a one-off business rule (the pure-pandas mirror is the
-- `tmin_gt_tmax` rule in weatherpipe.validate). The test passes when this query
-- returns zero rows (dbt's convention).

select
    station,
    date,
    tmin_c,
    tmax_c
from {{ ref('daily_weather') }}
where tmin_c > tmax_c
