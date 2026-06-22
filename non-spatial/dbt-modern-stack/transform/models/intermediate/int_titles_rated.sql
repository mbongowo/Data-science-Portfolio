-- Intermediate: join titles to their ratings and keep only titles that have a
-- rating. This is the grain the marts build on. An inner join here means the
-- fact never contains a rating without a title, and never a title with no
-- rating; the relationships test on the mart then guards the join key.

with titles as (

    select * from {{ ref('stg_titles') }}

),

ratings as (

    select * from {{ ref('stg_ratings') }}

)

select
    t.title_id,
    t.title_type,
    t.primary_title,
    t.start_year,
    t.runtime_minutes,
    t.genres,
    r.average_rating,
    r.num_votes
from titles t
inner join ratings r
    on t.title_id = r.title_id
