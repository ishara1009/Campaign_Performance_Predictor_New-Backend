-- ============================================================
-- Supabase – predictions table
-- Run this SQL in the Supabase SQL Editor:
--   https://yylwxfrhallymrtjjuuk.supabase.co
-- ============================================================

create table if not exists predictions (
  id                   uuid default gen_random_uuid() primary key,
  caption              text,
  content              text,
  platform             text not null,
  post_date            date not null,
  post_time            time not null,
  followers            integer not null,
  ad_boost             integer not null default 0,
  likes                numeric,
  comments             numeric,
  shares               numeric,
  clicks               numeric,
  timing_quality_score numeric,
  created_at           timestamptz default now()
);

-- Enable Row Level Security (optional – service_role bypasses it)
alter table predictions enable row level security;

-- Allow anon reads (for history page)
create policy "Allow anon read"
  on predictions for select
  using (true);

-- Allow service_role to insert/delete (backend uses service_role key)
create policy "Allow service_role all"
  on predictions for all
  using (auth.role() = 'service_role');
