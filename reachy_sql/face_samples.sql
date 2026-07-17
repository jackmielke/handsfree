-- Upgrade Wonder's face memory from "one photo per person" to a small face
-- model: each person accumulates several embedding samples over time (taken
-- automatically as they're recognized, at different angles/lighting), and
-- matching is nearest-sample-across-everyone rather than nearest-single-photo.
-- This is the fix for "it doesn't recognize me reliably" — one reference
-- photo is fragile to angle/lighting; several samples per person is not.
--
-- Run this in the Supabase SQL editor on the same project as faces.sql.

create table if not exists public.face_samples (
    id          uuid primary key default uuid_generate_v4(),
    face_id     uuid not null references public.faces(id) on delete cascade,
    embedding   double precision[] not null,   -- 128-d face encoding
    snapshot    text,                          -- data:image/jpeg;base64,... for this sample
    created_at  timestamptz not null default now()
);

create index if not exists face_samples_face_id_idx on public.face_samples(face_id);

alter table public.face_samples disable row level security;

-- Migrate each existing person's single embedding into a first sample, so
-- matching can move entirely onto face_samples going forward.
insert into public.face_samples (face_id, embedding, snapshot, created_at)
select id, embedding, snapshot, first_seen
from public.faces
where embedding is not null
  and not exists (
    select 1 from public.face_samples fs where fs.face_id = faces.id
  );

-- The identity row no longer needs to carry its own embedding; new rows
-- won't set it. Existing values are left in place (harmless) but the app
-- reads/writes samples exclusively via face_samples now.
alter table public.faces alter column embedding drop not null;

comment on table public.face_samples is
    'Multiple face-embedding samples per person, for more reliable recognition than a single reference photo.';
