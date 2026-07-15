-- Face memory for Wonder (the Reachy Mini).
-- Run this in the Supabase SQL editor (or as a migration) on the project you
-- want Wonder to remember people on — e.g. the Vibe community project.
--
-- Design: each row is one known person. `embedding` is a 128-float face
-- encoding (from the `face_recognition` library) used to recognize them again.
-- `snapshot` is a small JPEG data URI of the first sighting, so the dashboard
-- can show a face gallery. Recognition/matching happens client-side in
-- reachy_memory.py (nearest-neighbour over embeddings).

create extension if not exists "uuid-ossp";

create table if not exists public.faces (
    id           uuid primary key default uuid_generate_v4(),
    name         text,                          -- null until you name them
    embedding    double precision[] not null,   -- 128-d face encoding
    snapshot     text,                          -- data:image/jpeg;base64,... (first sighting)
    times_seen   integer not null default 1,
    first_seen   timestamptz not null default now(),
    last_seen    timestamptz not null default now(),
    notes        text
);

-- Wonder reads/writes with the service key, so RLS can stay off for now.
-- If you later expose this to the browser with the anon key, add policies.
alter table public.faces disable row level security;

comment on table public.faces is 'People Wonder (Reachy Mini) has seen and can recognize.';
