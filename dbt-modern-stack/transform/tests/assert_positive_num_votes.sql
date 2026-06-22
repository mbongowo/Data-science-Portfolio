-- Singular test: a rated title should never have a non-positive vote count.
-- A generic test cannot express "> 0" without a package macro, so this is the
-- natural place for a one-off business rule. The test passes when this query
-- returns zero rows (dbt's convention).

select
    title_id,
    num_votes
from {{ ref('fct_title_rating') }}
where num_votes <= 0
