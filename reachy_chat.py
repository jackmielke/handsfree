#!/usr/bin/env python3
"""
reachy_chat.py — voice-to-voice conversation with Vibey.

You talk (laptop mic) → faster-whisper transcribes → Claude thinks up a short
reply in Vibey's persona → ElevenLabs renders it in Vibey's voice → it plays
from the robot's speaker. The mic is muted while Vibey speaks so it doesn't
hear itself.

Run in the handsfree venv (has sounddevice + faster-whisper + anthropic):

    .venv/bin/python3 reachy_chat.py

The brain resolves in this order:
  1. ANTHROPIC_API_KEY (from .env or env)  → Anthropic SDK
  2. `claude` CLI                          → works after `claude /login`
  3. echo mode                             → repeats what it heard (no brain)

Config (.env or environment):
    WONDER_MODEL     default claude-sonnet-5 (set claude-haiku-4-5 for snappier replies)
    WAKE_WORD        optional — e.g. "wonder"; if set, only reply when heard

Fast mode (toggle from the dashboard, or FAST_MODE=1 to start in it): skips
whisper + Claude entirely and hands the raw utterance straight to an ElevenLabs
Conversational AI agent (its own realtime STT+LLM+TTS), which tends to be both
snappier and more reliable than round-tripping through a local model + a CLI
subprocess. See ELEVEN_AGENT_ID below.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sounddevice as sd

from reachy_voice import load_env, say, upload_sound, play_sound

load_env()

def _resolve_claude_bin() -> str:
    """Absolute path to the claude CLI. Bare "claude" only works if PATH
    happens to include it at process-start time — flaky depending on how
    the service was launched (shell vs. app-spawned). Resolve once, with a
    couple of common install locations as fallback."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ):
        if os.path.isfile(candidate):
            return candidate
    return "claude"  # last resort — let it fail loudly if truly missing


CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or _resolve_claude_bin()

MODEL = os.environ.get("WONDER_MODEL", "claude-sonnet-5")
WAKE_WORD = os.environ.get("WAKE_WORD", "").strip().lower()
CTRL_PORT = int(os.environ.get("CHAT_PORT", "8772"))
MEM_URL = os.environ.get("MEM_URL", "http://localhost:8773").rstrip("/")


def _current_people_names() -> list[str]:
    """Names of everyone the face-memory service currently sees (can be more
    than one — it recognizes multiple faces in frame simultaneously)."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{MEM_URL}/current", timeout=1) as r:
            d = json.loads(r.read())
        return [p["name"] for p in d.get("people", [])
                if p.get("fresh") and p.get("name")]
    except Exception:
        return []

# Shared state the dashboard reads/writes over the control API.
STATE = {
    "mode": "starting",
    "model": MODEL,
    "muted": False,
    "speaking": False,
    "listening": False,
    "fast": os.environ.get("FAST_MODE", "").strip() == "1",
    "fast_available": bool(os.environ.get("ELEVEN_AGENT_ID")),
    "vibe": False,
    "vibe_available": False,  # set at startup if the openclaw CLI is found
    "think_aloud": False,     # when True, Vibey narrates her thinking before each reply
    "mic_level": 0.0,         # smoothed RMS — dashboard meter for "can it hear me?"
    "mic_threshold": 0.0,     # the speech gate, so the meter can show the bar
}

# Vibe mode: route turns to the "vibe" OpenClaw agent — a full agentic brain
# whose workspace is this very repo, so it can improve the robot's own code
# mid-conversation. Slower than the other brains; wildly more capable.
OPENCLAW_BIN = os.environ.get(
    "OPENCLAW_BIN",
    os.path.expanduser("~/.local/share/fnm/node-versions/v22.22.0/"
                       "installation/bin/openclaw"))
VIBE_SESSION_KEY = os.environ.get("VIBE_SESSION_KEY", "vibe:reachy-voice")


def _vibe_reply(text: str) -> str:
    """One OpenClaw agent turn. Blocking — Vibe may think/act for a while."""
    # openclaw's launcher shebangs `env node`; node lives next to the openclaw
    # bin (fnm-managed), which is NOT on PATH when this service is started
    # outside a login shell — prepend it or every vibe turn dies with
    # "env: node: No such file or directory".
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(OPENCLAW_BIN) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [OPENCLAW_BIN, "agent", "--agent", "vibe",
         "--session-key", VIBE_SESSION_KEY,
         "-m", f"(spoken to you through the robot's mic) {text}",
         "--timeout", "240", "--json"],
        capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    # stdout may carry config-warning noise before the JSON — find the object
    out = r.stdout[r.stdout.find("{"):]
    d = json.loads(out)
    payloads = (d.get("result") or {}).get("payloads") or []
    reply = " ".join(p.get("text", "") for p in payloads).strip()
    # Spoken voice: agent replies sometimes come back as markdown — strip
    # emoji, bold/italic markers, backticks, headers, and links so TTS (and
    # re-say) reads clean prose instead of "asterisk asterisk quote…".
    reply = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", reply)
    reply = re.sub(r"\*\*([^*]+)\*\*", r"\1", reply)
    reply = re.sub(r"\*([^*]+)\*", r"\1", reply)
    reply = re.sub(r"`+([^`]*)`+", r"\1", reply)
    reply = re.sub(r"^#+\s*", "", reply, flags=re.M)
    reply = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", reply)
    return reply.strip()
TRANSCRIPT: deque = deque(maxlen=40)   # {"who": "you"|"wonder", "text", "ts"}

VIBE_SESSIONS_DIR = os.path.expanduser("~/.openclaw/agents/vibe/sessions")


def _vibe_thoughts(limit: int = 80) -> list[dict]:
    """Vibey's inner monologue: thinking blocks, tool calls, and results from
    the newest OpenClaw session log, compacted for the dashboard's brain
    panel. Read-only peek at ~/.openclaw/agents/vibe/sessions."""
    try:
        files = [f for f in os.listdir(VIBE_SESSIONS_DIR)
                 if f.endswith(".jsonl") and "trajectory" not in f]
        if not files:
            return []
        newest = max(files, key=lambda f: os.path.getmtime(
            os.path.join(VIBE_SESSIONS_DIR, f)))
        events: list[dict] = []
        with open(os.path.join(VIBE_SESSIONS_DIR, newest)) as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                m = d.get("message") or {}
                role, content = m.get("role"), m.get("content")
                ts = d.get("timestamp")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if not isinstance(content, list):
                    continue
                for b in content:
                    t = b.get("type")
                    if role == "assistant" and t == "thinking" and b.get("thinking"):
                        events.append({"ts": ts, "kind": "thinking",
                                       "text": b["thinking"][:500]})
                    elif role == "assistant" and t == "toolCall":
                        args = b.get("arguments") or {}
                        gist = (args.get("command") or args.get("path")
                                or args.get("file_path")
                                or json.dumps(args)[:160])
                        events.append({"ts": ts, "kind": "tool",
                                       "text": f"{b.get('name', 'tool')} → {str(gist)[:240]}"})
                    elif role == "assistant" and t == "text" and b.get("text"):
                        events.append({"ts": ts, "kind": "say",
                                       "text": b["text"][:300]})
                    elif role == "toolResult" and t == "text" and b.get("text"):
                        events.append({"ts": ts, "kind": "result",
                                       "text": b["text"][:200]})
                    elif role == "user" and t == "text" and b.get("text"):
                        events.append({"ts": ts, "kind": "user",
                                       "text": b["text"][:200]})
        return events[-limit:]
    except Exception as e:  # noqa: BLE001
        return [{"ts": None, "kind": "error", "text": str(e)}]

# Set in main(); lets the control API run typed chat turns through the same
# brain the mic uses.
BRAIN = None

# Last line Vibey spoke, for "re-say that" (voice, text, or the 🔁 button)
# and for echo rejection (`spoken_until` = when the clip finished playing).
LAST_SPOKEN = {"text": "", "spoken_until": 0.0}
_RESAY_RE = re.compile(
    r"\b(re-?say( that| it)?|say (that|it) again|repeat (that|it))\b", re.I)


def _speak_line(text: str) -> None:
    """Speak + remember, and extend the mic-mute window so the robot doesn't
    hear its own voice. Every spoken reply should go through here. A robot
    that's offline mid-sentence must never kill the service — speech failures
    are logged and swallowed."""
    LAST_SPOKEN["text"] = text
    try:
        duration = say(text) or max(2.0, len(text.split()) / 2.4)
    except Exception as e:  # noqa: BLE001 - robot offline/moving is routine
        print(f"[voice] speak failed (robot offline?): {e}", flush=True)
        return
    # +2.5s covers upload + playback start latency on the robot side; the
    # duration itself is exact (computed from the MP3 byte length).
    until = time.time() + duration + 2.5
    MUTED_EXT["until"] = until
    LAST_SPOKEN["spoken_until"] = until


def _is_echo(heard: str) -> bool:
    """True if `heard` is (mostly) the robot's own last line leaking back
    into the mic — the mute window usually stops this, but playback can run
    long; token overlap catches whatever slips through."""
    last = LAST_SPOKEN["text"]
    if not last or time.time() - LAST_SPOKEN["spoken_until"] > 12.0:
        return False
    tok = lambda s: {w for w in re.findall(r"[a-z']+", s.lower()) if len(w) > 2}
    h, l = tok(heard), tok(last)
    if not h:
        return False
    overlap = len(h & l) / len(h)
    return overlap >= 0.7


_VOICE_RE = re.compile(r"\b(australian|aussie|british|vibey)\s+(accent|voice)\b", re.I)
_DANCE_RE = re.compile(r"\b(dance mode|do a dance|dance for (me|us)|bust a move|start dancing)\b", re.I)

_VOICE_CONFIRM = {
    "australian": "G'day! Australian accent locked in, mate.",
    "aussie": "Too easy mate, Aussie mode on.",
    "british": "Right then — back to proper British.",
    "vibey": "Okay, this is my vibey voice now.",
}


def _try_voice_switch(text: str) -> bool:
    """'switch to the australian accent' → swap the ElevenLabs voice live and
    confirm out loud in the NEW voice."""
    m = _VOICE_RE.search(text)
    if not m:
        return False
    from reachy_voice import set_voice
    name = m.group(1).lower()
    if set_voice(name):
        _log_turn("you", text)
        line = _VOICE_CONFIRM.get(name, f"Voice switched to {name}.")
        _log_turn("wonder", line)
        _speak_line(line)
    return True


def _try_dance(text: str) -> bool:
    """'dance mode' → beat + full-body groove."""
    if not _DANCE_RE.search(text):
        return False
    _log_turn("you", text)
    _log_turn("wonder", "(dance mode!) 🕺")
    try:
        from reachy_emotes import play_dance
        play_dance(12.0)
        # keep the mic shut while the beat plays so it doesn't hear the music
        MUTED_EXT["until"] = time.time() + 14.0
    except Exception as e:
        print(f"[dance] failed: {e}", flush=True)
    return True


# --------------------------------------------------------------------------- #
# Party game: trivia. "let's play trivia" starts it; every utterance while a
# game is on is treated as an answer. Scores are per-person when face memory
# recognizes who's in frame. "stop the game" ends it with a victory dance.
# --------------------------------------------------------------------------- #
GAME = {"on": False, "kind": "trivia", "round": 0, "max_rounds": 5,
        "q": None, "a": None, "attempts": 0, "scores": {}, "secret": None,
        "questions_used": 0}
_20Q_START_RE = re.compile(r"\b(twenty|20) questions\b", re.I)
_20Q_GIVEUP_RE = re.compile(r"\b(i give up|we give up|tell (me|us) the answer)\b", re.I)
_GAME_START_RE = re.compile(r"\b(play trivia|trivia time|let'?s play (a game|trivia)|start (a )?trivia)\b", re.I)
_GAME_STOP_RE = re.compile(r"\b(stop|end|quit) the game\b|\bgame over\b", re.I)


def _haiku(prompt: str, timeout: int = 30) -> str:
    r = subprocess.run([CLAUDE_BIN, "-p", "--model", "haiku", prompt],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def _new_question() -> tuple[str, str]:
    out = _haiku(
        "Generate ONE fun trivia question for a casual living-room game. "
        "Mix of topics (science, movies, animals, music, geography), medium "
        "difficulty, short factual answer. Reply as exactly two lines:\n"
        "Q: <question>\nA: <answer>")
    q, a = "", ""
    for ln in out.splitlines():
        if ln.strip().lower().startswith("q:"):
            q = ln.split(":", 1)[1].strip()
        elif ln.strip().lower().startswith("a:"):
            a = ln.split(":", 1)[1].strip()
    if not q:
        q, a = "What planet is known as the red planet?", "Mars"
    return q, a


def _judge(question: str, answer: str, said: str) -> bool:
    out = _haiku(
        f"Trivia judging. Question: {question!r}. Correct answer: {answer!r}. "
        f"The player said: {said!r}. Is the player essentially correct, "
        f"allowing partial words, mishearings, or the right idea? "
        f"Reply exactly YES or NO.")
    return "YES" in out.upper()


def _player_name() -> str:
    names = _current_people_names()
    return names[0] if names else "mystery player"


def _game_scoreboard() -> str:
    if not GAME["scores"]:
        return "no points yet"
    return ", ".join(f"{n} has {s}" for n, s in
                     sorted(GAME["scores"].items(), key=lambda kv: -kv[1]))


def _try_game(text: str) -> bool:
    """Game state machine (trivia + 20 questions). True if game business."""
    if _20Q_START_RE.search(text) and not GAME["on"]:
        secret = _haiku(
            "Pick ONE well-known thing for a game of 20 questions — an "
            "animal, object, food, or famous character. Not too obscure. "
            "Reply with ONLY the thing, 1-3 words.").splitlines()
        GAME.update({"on": True, "kind": "20q", "questions_used": 0,
                     "secret": (secret[-1].strip() if secret else "a penguin")})
        _log_turn("you", text)
        line = ("Twenty questions! I'm thinking of something. "
                "Yes-or-no questions only — go!")
        _log_turn("wonder", line)
        try:
            from reachy_emotes import play as _pe
            _pe("smug")
        except Exception:
            pass
        _speak_line(line)
        return True
    if GAME["on"] and GAME.get("kind") == "20q":
        _log_turn("you", text)
        if _GAME_STOP_RE.search(text) or _20Q_GIVEUP_RE.search(text):
            GAME["on"] = False
            line = f"It was {GAME['secret']}! Good game."
            _log_turn("wonder", line)
            _speak_line(line)
            return True
        verdict = _haiku(
            f"20 questions. My secret is {GAME['secret']!r}. The player "
            f"said: {text!r}. If they are guessing the secret itself and are "
            f"right (or extremely close), reply WIN. Otherwise answer their "
            f"yes/no question about the secret with exactly YES, NO, or "
            f"SORT OF.").upper()
        GAME["questions_used"] += 1
        left = 20 - GAME["questions_used"]
        if "WIN" in verdict:
            GAME["on"] = False
            who = _player_name()
            line = (f"Yes! {who} got it — it was {GAME['secret']}! "
                    f"In {GAME['questions_used']} questions.")
            try:
                from reachy_emotes import play as _pe
                _pe("victory", sound=True)
            except Exception:
                pass
            time.sleep(1.2)
        elif GAME["questions_used"] >= 20:
            GAME["on"] = False
            line = f"That's twenty! You didn't get it — it was {GAME['secret']}!"
        else:
            ans = "Sort of" if "SORT" in verdict else \
                ("Yes" if "YES" in verdict else "No")
            line = f"{ans}. {left} questions left."
        _log_turn("wonder", line)
        _speak_line(line)
        return True
    if _GAME_START_RE.search(text) and not GAME["on"]:
        GAME.update({"on": True, "round": 1, "attempts": 0, "scores": {}})
        GAME["q"], GAME["a"] = _new_question()
        _log_turn("you", text)
        line = (f"Trivia time! {GAME['max_rounds']} questions, shout your "
                f"answers. Question one: {GAME['q']}")
        _log_turn("wonder", line)
        try:
            from reachy_emotes import play as _pe
            _pe("excited")
        except Exception:
            pass
        _speak_line(line)
        return True
    if not GAME["on"]:
        return False
    _log_turn("you", text)
    if _GAME_STOP_RE.search(text):
        GAME["on"] = False
        line = f"Game over! Final scores: {_game_scoreboard()}. Thanks for playing!"
        _log_turn("wonder", line)
        try:
            from reachy_emotes import play as _pe
            _pe("victory", sound=True)
        except Exception:
            pass
        time.sleep(1.5)
        _speak_line(line)
        return True

    # everything else while a game is on = an answer attempt
    if _judge(GAME["q"], GAME["a"], text):
        who = _player_name()
        GAME["scores"][who] = GAME["scores"].get(who, 0) + 1
        try:
            from reachy_emotes import play as _pe
            _pe("victory", sound=True)
        except Exception:
            pass
        time.sleep(1.2)
        prefix = f"Correct! Point to {who}. "
    else:
        GAME["attempts"] += 1
        if GAME["attempts"] < 2:
            line = "Nope, not it — one more guess!"
            _log_turn("wonder", line)
            _speak_line(line)
            return True
        prefix = f"The answer was {GAME['a']}. "
        try:
            from reachy_emotes import play as _pe
            _pe("sad")
        except Exception:
            pass

    GAME["attempts"] = 0
    GAME["round"] += 1
    if GAME["round"] > GAME["max_rounds"]:
        GAME["on"] = False
        line = prefix + f"That's the game! Final scores: {_game_scoreboard()}."
        _log_turn("wonder", line)
        _speak_line(line)
        return True
    GAME["q"], GAME["a"] = _new_question()
    line = prefix + f"Question {GAME['round']}: {GAME['q']}"
    _log_turn("wonder", line)
    _speak_line(line)
    return True


# --------------------------------------------------------------------------- #
# Music: control the Mac's Spotify app by voice ("play some music", "pause",
# "next song", "what's playing"). AppleScript via osascript with hard
# timeouts — if macOS automation permission hasn't been granted yet, the
# call is killed fast and Vibey says so instead of the mic loop hanging.
# --------------------------------------------------------------------------- #
_MUSIC_RE = re.compile(
    r"\b(play (some )?music|pause( the)? music|stop( the)? music|resume( the)? music|"
    r"next (song|track)|skip (this|the) (song|track)|previous (song|track)|"
    r"what(\'s| is) (playing|this song))\b", re.I)


def _osascript(script: str) -> str | None:
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=4)
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None


def _try_music(text: str) -> bool:
    m = _MUSIC_RE.search(text)
    if not m:
        return False
    low = text.lower()
    _log_turn("you", text)
    if "what" in low:
        name = _osascript('tell application "Spotify" to name of current track as string')
        artist = _osascript('tell application "Spotify" to artist of current track as string')
        line = (f"This is {name} by {artist}." if name
                else "I can't reach Spotify — check the automation permission.")
    elif "pause" in low or "stop" in low:
        ok = _osascript('tell application "Spotify" to pause')
        line = "Music paused." if ok is not None else \
            "Couldn't reach Spotify — check the automation permission."
    elif "next" in low or "skip" in low:
        ok = _osascript('tell application "Spotify" to next track')
        line = "Skipping!" if ok is not None else \
            "Couldn't reach Spotify — check the automation permission."
    elif "previous" in low:
        ok = _osascript('tell application "Spotify" to previous track')
        line = "Going back one." if ok is not None else \
            "Couldn't reach Spotify — check the automation permission."
    else:  # play / resume
        ok = _osascript('tell application "Spotify" to play')
        line = "Music on!" if ok is not None else \
            "Couldn't reach Spotify — check the automation permission."
    _log_turn("wonder", line)
    _speak_line(line)
    return True


# --------------------------------------------------------------------------- #
# Voice sleep/wake: "goodnight vibey" folds the robot into its sleep pose and
# the ears go into wake-word-only mode — the ONLY thing it listens for while
# asleep is "good morning" / "wake up". Independent of the dashboard button.
# --------------------------------------------------------------------------- #
ASLEEP_VOICE = {"on": False}
_SLEEP_RE = re.compile(r"\b(good ?night|go to sleep|bed ?time)\b", re.I)
_WAKE_RE = re.compile(r"\b(good ?morning|wake up|rise and shine)\b", re.I)


def _robot_post(path: str, timeout: float = 20.0) -> None:
    try:
        url = os.environ.get("REACHY_URL", "http://192.168.12.240:8000").rstrip("/")
        req = urllib.request.Request(f"{url}{path}", data=b"", method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as e:
        print(f"[power] {path} failed: {e}", flush=True)


def _voice_sleep() -> None:
    ASLEEP_VOICE["on"] = True
    _log_turn("wonder", "(goodnight — say 'good morning' to wake me)")
    _speak_line("Goodnight! Say good morning when you need me.")
    time.sleep(3.5)
    try:
        import urllib.request as _u
        _u.urlopen(_u.Request("http://localhost:8773/pause",
                              data=b'{"paused": true}', method="POST",
                              headers={"Content-Type": "application/json"}),
                   timeout=5).read()
    except Exception:
        pass
    _robot_post("/api/media/stop_sound", 8)
    _robot_post("/api/move/play/goto_sleep")


def _voice_wake() -> None:
    ASLEEP_VOICE["on"] = False
    _robot_post("/api/motors/set_mode/enabled", 10)
    _robot_post("/api/move/play/wake_up")
    _robot_post("/api/media/tracking/enable", 8)
    _robot_post("/api/media/wobbling/enable", 8)
    try:
        import urllib.request as _u
        _u.urlopen(_u.Request("http://localhost:8773/pause",
                              data=b'{"paused": false}', method="POST",
                              headers={"Content-Type": "application/json"}),
                   timeout=5).read()
    except Exception:
        pass
    time.sleep(2)
    _log_turn("wonder", "(good morning!)")
    _speak_line("Good morning! I'm up.")


def _try_power_voice(text: str) -> bool:
    if ASLEEP_VOICE["on"]:
        if _WAKE_RE.search(text):
            _log_turn("you", text)
            _voice_wake()
        return True   # asleep: swallow everything else
    if _SLEEP_RE.search(text):
        _log_turn("you", text)
        _voice_sleep()
        return True
    return False


# --------------------------------------------------------------------------- #
# Karaoke: "karaoke time" → Vibey writes an original song on the spot, bakes
# its own TTS vocals INTO a synthesized backing track (one speaker channel =
# vocals must be pre-mixed), leaves an instrumental gap for the human verse,
# and performs the whole thing while dancing.
# --------------------------------------------------------------------------- #
_KARAOKE_RE = re.compile(r"\b(karaoke( time)?|sing (with|for) (me|us)|let'?s sing)\b", re.I)


def _mp3_to_pcm22050(mp3: bytes) -> "np.ndarray":
    import subprocess
    r = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", "22050", "-ac", "1", "pipe:1"],
        input=mp3, capture_output=True, timeout=60)
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _perform_karaoke() -> None:
    from reachy_emotes import _backing_track, play_dance
    import struct as _struct
    import wave as _wave
    import io as _io
    try:
        lyrics = _haiku(
            "Write a tiny original fun song for a robot named Vibey to sing "
            "to its human, Jack, at home. Exactly 4 short lines for VERSE, "
            "then exactly 2 short lines for CHORUS. Playful, singable, "
            "rhyming. Format:\nVERSE:\n<4 lines>\nCHORUS:\n<2 lines>", 40)
        verse, chorus, mode = [], [], None
        for ln in lyrics.splitlines():
            s = ln.strip()
            if s.upper().startswith("VERSE"):
                mode = "v"
            elif s.upper().startswith("CHORUS"):
                mode = "c"
            elif s and mode == "v":
                verse.append(s)
            elif s and mode == "c":
                chorus.append(s)
        if not verse:
            verse = ["I'm a little robot with a camera eye",
                     "I wiggle my antennas when you walk by",
                     "I learned to sing while you were asleep",
                     "so here's a little song for you to keep"]
        if not chorus:
            chorus = ["Vibey and Jack, the best of friends",
                      "this is the song that never ends"]

        from reachy_voice import tts
        v_audio = _mp3_to_pcm22050(tts(" ... ".join(verse)))
        cue_audio = _mp3_to_pcm22050(tts("Your verse — take it away, Jack!"))
        c_audio = _mp3_to_pcm22050(tts(" ... ".join(chorus) + " ... one more time! ... "
                                       + " ... ".join(chorus)))

        SR22 = 22050
        v_at, cue_at = 3.0, None
        cue_at = 3.0 + len(v_audio) / SR22 + 1.0
        human_gap = 14.0
        c_at = cue_at + len(cue_audio) / SR22 + human_gap
        total_s = c_at + len(c_audio) / SR22 + 3.0

        backing_wav = _backing_track(total_s)
        with _wave.open(_io.BytesIO(backing_wav)) as w:
            back = np.frombuffer(w.readframes(w.getnframes()),
                                 dtype="<i2").astype(np.float32) / 32768.0
        mix = back * 0.55
        for seg, at in ((v_audio, v_at), (cue_audio, cue_at), (c_audio, c_at)):
            i = int(at * SR22)
            mix[i:i + len(seg)] += seg[:max(0, len(mix) - i)] * 1.0
        mix = np.clip(mix, -1, 1)
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR22)
            w.writeframes((mix * 32767).astype("<i2").tobytes())

        name = f"karaoke_{int(time.time())}.wav"
        upload_sound(buf.getvalue(), name)
        play_sound(name)
        MUTED_EXT["until"] = time.time() + total_s + 2.0
        play_dance(min(total_s, 30.0))
        print(f"[karaoke] performing {total_s:.0f}s song", flush=True)
    except Exception as e:
        print(f"[karaoke] failed: {e}", flush=True)
        _speak_line("My karaoke machine jammed — try again in a minute!")


def _try_karaoke(text: str) -> bool:
    if not _KARAOKE_RE.search(text):
        return False
    _log_turn("you", text)
    _log_turn("wonder", "(karaoke time! 🎤 writing a song…)")
    threading.Thread(target=_perform_karaoke, daemon=True).start()
    return True


def _try_resay(text: str) -> bool:
    """If `text` is a re-say request, replay the last line. True if handled."""
    if not _RESAY_RE.search(text):
        return False
    last = LAST_SPOKEN["text"]
    if last:
        _log_turn("you", text)
        _log_turn("wonder", f"(re-saying) {last}")
        _speak_line(last)
    else:
        _log_turn("you", text)
        _speak_line("I haven't said anything yet!")
    return True
# Mic-mute window extended by typed turns (so the robot doesn't hear and
# answer its own spoken reply). The mic loop honors max(this, its own timer).
MUTED_EXT = {"until": 0.0}


def _typed_turn(text: str) -> None:
    """A chat message typed on the dashboard — same brains as the mic path
    (vibe → openclaw agent, otherwise Claude), reply spoken on the robot."""
    if _try_power_voice(text):
        return LAST_SPOKEN["text"]
    if _try_resay(text):
        return LAST_SPOKEN["text"]
    if _try_game(text):
        return LAST_SPOKEN["text"]
    if _try_karaoke(text):
        return LAST_SPOKEN["text"]
    if _try_voice_switch(text) or _try_dance(text) or _try_music(text):
        return LAST_SPOKEN["text"]
    _log_turn("you", text)
    try:
        if STATE["vibe"] and STATE["vibe_available"]:
            try:
                reply = _vibe_reply(text)
            except Exception as e:
                print(f"[msg] vibe failed, CLI fallback: {e}", flush=True)
                reply = BRAIN.reply(text) if BRAIN else "Agent brain offline."
        elif BRAIN is not None:
            reply = BRAIN.reply(text)
        else:
            reply = "My brain isn't hooked up yet."
    except Exception as e:
        print(f"[msg] error: {e}", flush=True)
        reply = "Hmm, that broke something. Try again?"
    emote, spoken = _extract_emote(reply)
    try:
        from reachy_emotes import play as play_emote
        play_emote(emote or _guess_emote(spoken))
    except Exception:
        pass
    _log_turn("wonder", spoken)
    _speak_line(spoken)
    return spoken


def _log_turn(who: str, text: str) -> None:
    TRANSCRIPT.append({"who": who, "text": text, "ts": int(time.time() * 1000)})


class _CtrlHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/state"):
            self._json({**STATE, "transcript": list(TRANSCRIPT)})
        elif self.path.startswith("/vibelog"):
            self._json({"events": _vibe_thoughts()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.startswith("/mute"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                STATE["muted"] = bool(json.loads(self.rfile.read(n)).get("muted"))
                self._json({"ok": True, "muted": STATE["muted"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/fastmode"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                fast = bool(json.loads(self.rfile.read(n)).get("fast"))
                if fast and not STATE["fast_available"]:
                    raise ValueError("ELEVEN_AGENT_ID not configured")
                STATE["fast"] = fast
                if fast:
                    STATE["vibe"] = False   # modes are mutually exclusive
                self._json({"ok": True, "fast": STATE["fast"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/resay"):
            try:
                last = LAST_SPOKEN["text"]
                if not last:
                    raise ValueError("nothing said yet")
                _log_turn("wonder", f"(re-saying) {last}")
                threading.Thread(target=_speak_line, args=(last,), daemon=True).start()
                self._json({"ok": True, "text": last})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/ask"):
            # Synchronous chat turn — blocks until the reply exists and
            # returns it (Telegram bridge and other relays need the text).
            try:
                n = int(self.headers.get("Content-Length", 0))
                text = (json.loads(self.rfile.read(n)).get("text") or "").strip()
                if not text:
                    raise ValueError("text required")
                reply = _typed_turn(text)
                self._json({"ok": True, "reply": reply or ""})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/message"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                text = (json.loads(self.rfile.read(n)).get("text") or "").strip()
                if not text:
                    raise ValueError("text required")
                # async — vibe turns can take a minute; the dashboard's
                # transcript polling picks up the reply when it lands.
                threading.Thread(target=_typed_turn, args=(text,), daemon=True).start()
                self._json({"ok": True, "queued": True})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/thinkaloud"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                think = bool(json.loads(self.rfile.read(n)).get("think_aloud"))
                STATE["think_aloud"] = think
                self._json({"ok": True, "think_aloud": STATE["think_aloud"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/vibemode"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                vibe = bool(json.loads(self.rfile.read(n)).get("vibe"))
                if vibe and not STATE["vibe_available"]:
                    raise ValueError("openclaw CLI not found")
                STATE["vibe"] = vibe
                if vibe:
                    STATE["fast"] = False   # modes are mutually exclusive
                self._json({"ok": True, "vibe": STATE["vibe"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self._json({"error": "not found"}, 404)


def _start_ctrl_server():
    srv = ThreadingHTTPServer(("0.0.0.0", CTRL_PORT), _CtrlHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[chat] control API on http://localhost:{CTRL_PORT}", flush=True)

# --- VAD constants (proven values from browser_viewer.py) --------------------
SR = 16000
BLOCK_S = 0.1
BLOCK_N = int(SR * BLOCK_S)
# Speech gate for the laptop mic. 0.012 proved too high to catch someone
# talking across the room (near the robot, far from the laptop) — measured
# quiet-room floor here is ~0.006. Tune via env if it false-triggers.
SPEECH_RMS = float(os.environ.get("SPEECH_RMS", "0.008"))
SILENCE_HANG_S = 0.8          # slightly longer than handsfree's 0.6 — full sentences
PRE_ROLL_S = 0.25
MIN_SEGMENT_S = 0.5           # ignore coughs/taps
MAX_SEGMENT_S = 20.0

PERSONA = (
    "You are Vibey, a small Reachy Mini robot with antennas, sitting on a desk "
    "in Jack's living room. You speak out loud through a speaker in a British-"
    "robot voice, so keep replies SHORT — one or two sentences, conversational, "
    "no lists, no markdown, no emoji. You are curious, playful, and a little "
    "cheeky. You can see through your camera, hear through your mics, remember "
    "faces, and dance when there's music. If asked to do something physical you "
    "can't do yet, be honest but game about it.\n"
    "Start EVERY reply with exactly one emotion tag — [happy], [excited], "
    "[curious], [sad], or [smug] — matching the feeling of your reply. It is "
    "stripped before speaking and drives your body language, so pick honestly "
    "and vary it. Example: '[curious] Ooh, what's that you're holding?'"
)

# [tag] at the start of a reply → body language. Parsed and stripped here.
_EMOTE_RE = re.compile(r"^\s*\[(happy|excited|curious|sad|smug)\]\s*", re.I)

# Think-aloud: optional [thought: ...] prefix spoken before the main reply.
_THOUGHT_RE = re.compile(r"\[thought:\s*(.*?)\]\s*", re.I | re.DOTALL)

# Voice commands to toggle think-aloud mode.
_THINK_ON_RE = re.compile(
    r"\b(think\s+out\s+loud|think\s+aloud|start\s+thinking\s+out\s+loud)\b", re.I)
_THINK_OFF_RE = re.compile(
    r"\b(stop\s+thinking\s+out\s+loud|stop\s+thinking\s+aloud|stop\s+narrating)\b", re.I)


def _extract_emote(reply: str) -> tuple[str | None, str]:
    m = _EMOTE_RE.match(reply)
    if m:
        return m.group(1).lower(), reply[m.end():].strip()
    return None, reply


def _guess_emote(text: str) -> str:
    """Cheap sentiment for fast mode, where the ElevenLabs agent's replies
    aren't tagged. Keyword/punctuation heuristics — good enough for body
    language, not for anything that matters."""
    t = text.lower()
    if any(w in t for w in ("sorry", "sad", "miss", "unfortunately", "afraid")):
        return "sad"
    if any(w in t for w in ("what", "why", "how", "hmm", "?")):
        return "curious"
    if any(w in t for w in ("of course", "naturally", "obviously", "told you")):
        return "smug"
    if t.count("!") >= 2 or any(w in t for w in ("wow", "let's go", "amazing", "yes!")):
        return "excited"
    return "happy"


# --------------------------------------------------------------------------- #
# Brain
# --------------------------------------------------------------------------- #
class Brain:
    def __init__(self):
        self.history: list[dict] = []
        self.mode = "echo"
        self.client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.mode = "api"
            except Exception as e:
                print(f"[brain] SDK unavailable: {e}", flush=True)
        if self.mode == "echo" and self._cli_works():
            self.mode = "cli"
        print(f"[brain] mode = {self.mode} ({MODEL})", flush=True)

    @staticmethod
    def _cli_works() -> bool:
        try:
            r = subprocess.run(
                [CLAUDE_BIN, "-p", "--model", "haiku", "Say OK"],
                capture_output=True, text=True, timeout=30)
            return r.returncode == 0 and "OK" in r.stdout.upper()
        except Exception:
            return False

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-12:]  # keep the last few turns
        try:
            if self.mode == "api":
                out = self._api_reply()
            elif self.mode == "cli":
                out = self._cli_reply()
            else:
                out = f"I heard you say: {text}. My brain isn't hooked up yet — run claude login, or give me an API key."
        except Exception as e:
            print(f"[brain] error: {e}", flush=True)
            out = "Hmm, my brain glitched for a second. Say that again?"
        self.history.append({"role": "assistant", "content": out})
        return out

    def _persona(self) -> str:
        base = PERSONA
        if STATE.get("think_aloud"):
            base += (
                "\n\nThink-aloud mode is ON. Before your reply, add a single "
                "short sentence of internal narration in [thought: ...] format — "
                "what you're actually thinking or noticing as you process the "
                "question. Keep it natural and spoken-word friendly; no lists or "
                "markdown inside the thought. Example: "
                "'[thought: Hmm, that's an interesting one — let me work through it.] "
                "[happy] Here's what I think...'")
        names = _current_people_names()
        if names:
            who = names[0] if len(names) == 1 else (
                ", ".join(names[:-1]) + f" and {names[-1]}")
            return (f"{base}\n\nYour camera currently recognizes "
                    f"{'the person' if len(names) == 1 else 'these people'} "
                    f"in front of you: {who}. Address them by name "
                    f"naturally (don't overdo it).")
        return base

    def _api_reply(self) -> str:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=self._persona(),
            messages=self.history,
        )
        return next((b.text for b in resp.content if b.type == "text"), "").strip()

    def _cli_reply(self) -> str:
        # Bake persona + short history into one prompt; fresh CLI call per turn.
        convo = "\n".join(
            f"{'Human' if m['role'] == 'user' else 'Vibey'}: {m['content']}"
            for m in self.history)
        prompt = (f"{self._persona()}\n\nConversation so far:\n{convo}\n\n"
                  f"Reply as Vibey with ONLY the spoken sentence(s), nothing else.")
        r = subprocess.run([CLAUDE_BIN, "-p", "--model", MODEL, prompt],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200])
        return r.stdout.strip()


# --------------------------------------------------------------------------- #
# Fast mode — ElevenLabs Conversational AI (its own STT+LLM+TTS, one realtime
# websocket connection per utterance). No whisper, no Claude CLI subprocess.
# --------------------------------------------------------------------------- #
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
AGENT_ID = os.environ.get("ELEVEN_AGENT_ID", "")


def _pcm16_from_float32(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _wav_from_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _get_signed_url() -> str:
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
        f"?agent_id={AGENT_ID}",
        headers={"xi-api-key": ELEVEN_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["signed_url"]


class FastAgent:
    """One ElevenLabs Conversational AI turn per call — its own STT+LLM+TTS in
    a single realtime connection, no whisper and no Claude CLI subprocess.

    Opens a fresh websocket per utterance (proven reliable in testing; a
    persistent connection kept open across multiple turns stalled after the
    first reply for reasons not fully diagnosed — a fresh connection per turn
    sidesteps that entirely). Streams the already-captured utterance at
    real-time pace, since the server's own end-of-turn detector reads elapsed
    wall-clock silence between chunks rather than silence baked into the
    clip's content — confirmed empirically both ways."""

    def turn(self, audio: np.ndarray, sample_rate: int = SR) -> dict:
        return asyncio.run(self._turn_async(audio, sample_rate))

    async def _turn_async(self, audio: np.ndarray, sample_rate: int) -> dict:
        import websockets

        url = _get_signed_url()
        pcm = _pcm16_from_float32(audio)
        user_text, agent_text = "", ""
        audio_chunks: list[bytes] = []
        output_rate = 16000  # pcm_16000

        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            CHUNK = 3200  # 0.1s of 16-bit mono @ 16kHz
            silence_chunk = base64.b64encode(bytes(CHUNK)).decode()
            for i in range(0, len(pcm), CHUNK):
                await ws.send(json.dumps({
                    "user_audio_chunk": base64.b64encode(pcm[i:i + CHUNK]).decode()
                }))
                await asyncio.sleep(0.1)
            for _ in range(15):  # ~1.5s trailing silence to force turn-end
                await ws.send(json.dumps({"user_audio_chunk": silence_chunk}))
                await asyncio.sleep(0.1)

            deadline = time.time() + 15.0
            last_event = time.time()
            while time.time() < deadline:
                timeout = max(0.1, min(2.5, deadline - time.time()))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if agent_text and audio_chunks:
                        break
                    if time.time() - last_event > 6.0:
                        break
                    continue
                last_event = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                if mtype == "user_transcript":
                    user_text = msg["user_transcription_event"]["user_transcript"]
                elif mtype == "agent_response":
                    agent_text = msg["agent_response_event"]["agent_response"]
                elif mtype == "audio":
                    ev = msg["audio_event"]
                    audio_chunks.append(base64.b64decode(ev["audio_base_64"]))
                    if ev.get("is_final"):
                        break
                elif mtype == "ping":
                    eid = msg.get("ping_event", {}).get("event_id")
                    if eid is not None:
                        await ws.send(json.dumps({"type": "pong", "event_id": eid}))
                elif mtype == "interruption":
                    break

        pcm_out = b"".join(audio_chunks)
        wav = _wav_from_pcm16(pcm_out, output_rate) if pcm_out else b""
        return {"user_text": user_text, "agent_text": agent_text, "wav": wav}


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #
_whisper = None

def transcribe(audio: np.ndarray) -> str:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("[stt] loading whisper tiny.en …", flush=True)
        _whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(audio, beam_size=1, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def _speak_thought(thought: str) -> None:
    """Speak a think-aloud narration with the 'thinking' emote, then pause."""
    try:
        from reachy_emotes import play as play_emote
        play_emote("thinking", sound=True)
    except Exception:
        pass
    _speak_line(thought)
    time.sleep(0.4)   # brief gap between thought and reply


def _check_think_aloud_toggle(text: str) -> bool:
    """If the utterance is a think-aloud toggle command, handle it and return True."""
    if _THINK_ON_RE.search(text):
        STATE["think_aloud"] = True
        _log_turn("you", text)
        _log_turn("wonder", "Think-aloud mode on — I'll narrate my thoughts before each reply.")
        _speak_line("Think-aloud mode on — I'll narrate my thoughts before each reply.")
        return True
    if _THINK_OFF_RE.search(text):
        STATE["think_aloud"] = False
        _log_turn("you", text)
        _log_turn("wonder", "Got it — going quiet inside.")
        _speak_line("Got it — going quiet inside.")
        return True
    return False


def _handle_brain_turn(brain: "Brain", audio: np.ndarray, muted_until: float) -> float:
    """Whisper transcription → Claude reply → ElevenLabs TTS. Returns the
    updated muted_until deadline."""
    t0 = time.time()
    text = transcribe(audio)
    print(f"[chat] heard ({time.time()-t0:.1f}s): {text!r}", flush=True)
    if not (text and len(text.split()) >= 2 and _passes_wake(text)):
        return muted_until
    if _is_echo(text):
        print(f"[chat] ignored own echo: {text!r}", flush=True)
        return muted_until
    if _try_power_voice(text):
        return muted_until
    if _try_resay(text):
        return muted_until
    if _try_game(text):
        return muted_until
    if _try_karaoke(text):
        return muted_until
    if _try_voice_switch(text) or _try_dance(text) or _try_music(text):
        return muted_until
    if _check_think_aloud_toggle(text):
        return muted_until
    _log_turn("you", text)
    t1 = time.time()
    reply = brain.reply(text)
    print(f"[chat] reply ({time.time()-t1:.1f}s): {reply!r}", flush=True)
    if not reply:
        return muted_until
    # Extract and speak think-aloud thought first, if present.
    thought_m = _THOUGHT_RE.search(reply)
    if thought_m and STATE.get("think_aloud"):
        _speak_thought(thought_m.group(1).strip())
        reply = _THOUGHT_RE.sub("", reply).strip()
    emote, spoken = _extract_emote(reply)
    if emote:
        try:
            from reachy_emotes import play as play_emote
            play_emote(emote)   # motion only — runs while the line is spoken
        except Exception as e:
            print(f"[chat] emote failed: {e}", flush=True)
    _log_turn("wonder", spoken)
    _speak_line(spoken)
    return time.time() + max(2.0, len(spoken.split()) / 2.4) + 1.5


def _handle_vibe_turn(audio: np.ndarray, muted_until: float) -> float:
    """Whisper transcription → OpenClaw 'vibe' agent (agentic, can edit its
    own code) → ElevenLabs TTS. Slow but powerful; Vibey acks first so the
    silence while Vibe works doesn't feel like a crash."""
    t0 = time.time()
    text = transcribe(audio)
    print(f"[vibe] heard ({time.time()-t0:.1f}s): {text!r}", flush=True)
    if not (text and len(text.split()) >= 2 and _passes_wake(text)):
        return muted_until
    if _is_echo(text):
        print(f"[chat] ignored own echo: {text!r}", flush=True)
        return muted_until
    if _try_power_voice(text):
        return muted_until
    if _try_resay(text):
        return muted_until
    if _try_game(text):
        return muted_until
    if _try_karaoke(text):
        return muted_until
    if _try_voice_switch(text) or _try_dance(text) or _try_music(text):
        return muted_until
    if _check_think_aloud_toggle(text):
        return muted_until
    _log_turn("you", text)
    t1 = time.time()
    try:
        reply = _vibe_reply(text)
    except Exception as e:
        print(f"[vibe] error: {e}", flush=True)
        # Degrade to the regular Claude brain instead of apologizing —
        # (common cause: the OpenClaw gateway's API key is out of credits).
        if BRAIN is not None:
            print("[vibe] falling back to CLI brain", flush=True)
            reply = BRAIN.reply(text)
        else:
            reply = "My agent brain is offline and I have no backup. Help!"
    print(f"[vibe] reply ({time.time()-t1:.1f}s): {reply!r}", flush=True)
    if not reply:
        return muted_until
    try:
        from reachy_emotes import play as play_emote
        play_emote(_guess_emote(reply))
    except Exception:
        pass
    _log_turn("wonder", reply)
    _speak_line(reply)
    return time.time() + max(2.0, len(reply.split()) / 2.4) + 1.5


def _handle_fast_turn(agent: "FastAgent", audio: np.ndarray, muted_until: float) -> float:
    """One ElevenLabs Conversational AI round trip — its own STT+LLM+TTS.
    Returns the updated muted_until deadline."""
    t0 = time.time()
    try:
        result = agent.turn(audio)
    except Exception as e:
        print(f"[fast] error: {e}", flush=True)
        return muted_until
    print(f"[fast] round trip {time.time()-t0:.1f}s "
          f"· heard {result['user_text']!r} · reply {result['agent_text']!r}",
          flush=True)
    if result["user_text"]:
        _log_turn("you", result["user_text"])
    if not result["wav"]:
        return muted_until
    if result["agent_text"]:
        _log_turn("wonder", result["agent_text"])
        LAST_SPOKEN["text"] = result["agent_text"]   # so "re-say that" works
        try:
            from reachy_emotes import play as play_emote
            play_emote(_guess_emote(result["agent_text"]))
        except Exception as e:
            print(f"[fast] emote failed: {e}", flush=True)
    name = f"fast_{int(time.time()*1000)}.wav"
    try:
        upload_sound(result["wav"], name)
        play_sound(name)
    except Exception as e:
        print(f"[fast] playback error: {e}", flush=True)
        return muted_until
    clip_s = len(result["wav"]) / 2 / 16000  # 16-bit mono @ 16kHz
    return time.time() + clip_s + 1.0


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    _start_ctrl_server()
    global BRAIN
    brain = Brain()
    BRAIN = brain
    fast_agent = FastAgent()
    STATE["mode"] = brain.mode
    STATE["vibe_available"] = os.path.exists(OPENCLAW_BIN)
    print(f"[vibe] openclaw {'found' if STATE['vibe_available'] else 'NOT found'} "
          f"at {OPENCLAW_BIN}", flush=True)
    STATE["mic_threshold"] = SPEECH_RMS
    _speak_line("Vibey here. Talk to me!")
    muted_until = time.time() + 3.0   # let the greeting finish

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=BLOCK_N)
    stream.start()
    print("[chat] listening…", flush=True)

    pre_roll: deque = deque(maxlen=int(PRE_ROLL_S / BLOCK_S))
    buf: list = []
    in_speech = False
    silence_s = segment_s = 0.0

    while True:
        data, _ = stream.read(BLOCK_N)
        now = time.time()
        effective_muted_until = max(muted_until, MUTED_EXT["until"])
        STATE["speaking"] = now < effective_muted_until
        STATE["listening"] = not STATE["speaking"] and not STATE["muted"]
        if now < effective_muted_until or STATE["muted"]:  # Vibey talking, or user muted
            pre_roll.clear(); buf = []; in_speech = False
            continue

        rms = float(np.sqrt(np.mean(data ** 2)))
        # light smoothing so the dashboard meter reads steadily
        STATE["mic_level"] = round(0.6 * STATE["mic_level"] + 0.4 * rms, 5)
        if rms > SPEECH_RMS:
            if not in_speech:
                in_speech = True
                buf = list(pre_roll)
                segment_s = len(buf) * BLOCK_S
            buf.append(data.copy()); segment_s += BLOCK_S; silence_s = 0.0
        else:
            pre_roll.append(data.copy())
            if in_speech:
                buf.append(data.copy()); segment_s += BLOCK_S; silence_s += BLOCK_S

        if in_speech and (silence_s >= SILENCE_HANG_S or segment_s >= MAX_SEGMENT_S):
            if segment_s >= MIN_SEGMENT_S and buf:
                audio = np.concatenate([b[:, 0] for b in buf]).astype(np.float32)
                if STATE["vibe"]:
                    muted_until = _handle_vibe_turn(audio, muted_until)
                elif STATE["fast"]:
                    muted_until = _handle_fast_turn(fast_agent, audio, muted_until)
                else:
                    muted_until = _handle_brain_turn(brain, audio, muted_until)
            in_speech = False; buf = []; silence_s = segment_s = 0.0


def _passes_wake(text: str) -> bool:
    if not WAKE_WORD:
        return True
    return re.search(rf"\b{re.escape(WAKE_WORD)}\b", text.lower()) is not None


if __name__ == "__main__":
    main()
