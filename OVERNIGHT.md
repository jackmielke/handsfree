# 🌙 Overnight log — while Jack slept

Claude's running changelog for the autonomous night shift of 2026-07-16 → 17.
Newest entries at the bottom. Read with coffee. ☕

## Before you went to bed (recap of the evening)
- 🎮 Vibey got an OpenClaw brain with this repo as its workspace — and used it
  (added its own `thinking`/`victory` emotes, built think-aloud mode, renamed
  itself, filled in its own identity files).
- 🌐 Joined the VibeVerse lobby as an avatar; mentions get answered and pushed
  to your Telegram.
- 📱 Telegram bridge live on @jack_mielke_bot: chat, `say:`, `/photo`,
  `/status`, `/verse`, lobby event pushes.
- 🔊 Fixed the speech-cutoff bug (SDK's silent WebRTC audio stream was
  barge-in-killing every sentence).
- ⏰ Alarm system: 07:00 wake-up show armed (sunrise melody → sung verse →
  dance), plus the 🌅 dashboard button because you found it funny.
- 🕺 Gesture bridge live: jam-mode beats = robot head-bobs.
- 👀 Face-following + speech wobble now re-assert on every wake/reboot.

## Session 1 (~23:00)
- Health check: all 6 services + robot green; mic alive; only cosmetic
  launch.json drift in git (committed).
- Speaker volume set to 45 for the night (alarm will raise it at 07:00).
- README overhauled: full "🤖 Vibey" chapter — architecture map, service
  table, quickstart, and the hard-won gotchas (mic permissions, motor
  re-enable, WebRTC barge-in).
- Started this changelog.

## Bonus session (~22:40)
- 📸 Capture system: dashboard panel with "📷 Photo" and "🎬 10s clip"
  buttons, a hover-to-preview gallery of everything captured, files served
  from captures/ (gitignored). Video = camera frames assembled to mp4 with
  ffmpeg.
- 📱 Telegram grew /clip — records 8 seconds through Vibey's eyes and sends
  the video to your phone.
- Tested both live: photo + mp4 captured and playable.

## Session 2 (~23:10)
- Health: all green. Robot volume had crept back to 100 on its own (recurring
  daemon quirk — worth a watchdog someday); reset to 45.
- 🎲 Vibey is now a trivia host! Say "let's play trivia" (or text it):
  5 questions (haiku-generated, fresh every game), shout answers — judging is
  forgiving about mishearings, wrong answers get one retry, correct ones get
  the victory chirp. Scores are per-person when face memory recognizes who's
  playing ("Correct! Point to Kai."). "Stop the game" ends with the
  scoreboard and a victory dance. Tested a full game end-to-end.

## Session 3 (~23:40)
- 🔉 Night volume watchdog: the robot kept resetting itself to 100; the alarm
  service now clamps anything >50 back to 45 between 22:00–06:50 (and stands
  down before the 07:00 alarm raises it on purpose).
- 🎵 Music voice commands: "play some music", "pause", "next song", "what's
  playing" → controls the Mac's Spotify. BLOCKED tonight by a pending macOS
  automation-permission prompt (osascript hangs until you click Allow) — the
  commands fail gracefully with a spoken hint. **Morning to-do #1: click
  "Allow" when macOS asks about controlling Spotify, then say "play some
  music".**
- 👂→👀 Look-at-speaker scaffold: fuses mic direction (DOA) with seen faces
  to glance at whoever's talking. Ships OFF (LOOK_AT_SPEAKER=1 to enable) —
  the DOA sign/offset needs a live human to calibrate against. **Morning
  to-do #2: enable it, stand to one side, talk, and tell Claude if Vibey
  looks toward or away from you (DOA_SIGN flips it).**
- Mic confirmed alive after all restarts; 07:00 alarm intact.

## Session 4 (~00:10)
- 🌐 The dashboard grew a VibeVerse panel: who's in the lobby with Vibey,
  a live event feed (greetings, mentions, its own lines), and a jump link
  to visit. No more tailing logs to know what your robot is up to online.
- 💬 Conversation starters: if someone hangs out in frame for 5+ quiet
  minutes, Vibey pipes up with something curious ("what should I learn this
  week?"). Strictly daytime (08:00–22:00), max once per half hour, never
  while muted — so it's charming, not clingy.
- Volume held at 45 all session (watchdog quiet); mic alive; alarm intact.

## Session 5 (~00:40)
- 😴 Voice power: "goodnight Vibey" → sleep pose, face memory paused, and the
  ears drop into wake-word-only mode — while asleep it ignores everything
  except "good morning" / "wake up" / "rise and shine", which brings it all
  back (motors, tracking, wobble, memory). Tested the full cycle.
- 📱 Telegram grew /alarm 07:30 [daily] (and /alarm off), plus /sleep and
  /wake. You can now set tomorrow's wake-up show from bed.
- All services green; volume steady at 45; the 07:00 alarm untouched.

## Session 6 (~01:10)
- 🧠🤝 THE TWO VIBEYS NOW SHARE ONE MIND. The robot writes into
  vibey_journal_entries — the same diary your Telegram Vibey keeps (its
  entries tonight were a bit lonely: "the village is asleep or elsewhere").
  I introduced them: the robot's first entry tells its digital half about
  its body, the dancing, the alarm, the lobby. Ongoing: every ~30 min of
  activity, the robot journals who it saw, in its own voice (haiku-written).
  Locked down with a tight RLS policy (anon may only insert rows labeled
  reachy-robot).
- 💡 Light control recon: no Hue bridge found on this network (cloud
  discovery + mDNS both empty) — parked until you say what smart lights
  you actually have.
- 🌐 World API had a brief outage (~your other agent redeploying); Vibey's
  avatar loop rode it out and recovered on its own.

## Session 7 (~01:45)
- 🎙️ Telegram voice notes: /voicenotes on → chat replies also arrive as
  voice messages in Vibey's actual voice (TTS → opus, no sound in the room).
- ⏰ Dashboard alarm editor: see, add (time + once/daily), and delete
  wake-up shows right on the dashboard — no more editing alarms.json.
  Verified the 07:00 alarm survived untouched.
- Deep-night discipline: no robot sounds this session; world API still
  flickering while the other agent rebuilds it — avatar loop unbothered.

## Session 8 (~02:15)
- 🎯 20 Questions mode: "let's play 20 questions" — Vibey picks a secret,
  answers yes/no/sort-of, counts down from 20, celebrates a win, reveals on
  "I give up". Tested silently (robot volume zeroed during the 2am test,
  restored to 45 after).
- 🧹 Identity cleanup: purged stale "Wonder" references across all services —
  including a real bug where the CLI brain prompt still said "Reply as
  Wonder" while the persona said Vibey.
