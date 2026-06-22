-- Mart: a small aggregate the BI dashboard reads directly. One row per
-- title_type, summarising the rating fact: how many rated titles, the mean
-- rating, the total votes, and the share of "high"-band titles. Keeping the
-- aggregate as its own modelled, tested table means the dashboard never runs
-- ad-hoc GROUP BYs against the fact and the numbers are covered by tests.

with rated as (

    select * from {{ ref('fct_title_rating') }}

)

select
    title_type,
    count(*)                                              as num_rated_titles,
    round(avg(average_rating), 3)                         as avg_rating,
    sum(num_votes)                                        as total_votes,
    round(
        avg(case when rating_band = 'high' then 1.0 else 0.0 end), 3
    )                                                     as high_band_share
from rated
group by title_type
