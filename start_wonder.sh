#!/bin/zsh
# start_wonder.sh — boot the whole Wonder stack in one command.
#
#   ./start_wonder.sh          start everything (idempotent: restarts cleanly)
#   ./start_wonder.sh stop     stop everything
#
# Services:
#   :8771  reachy_camera.py  — robot camera → MJPEG   (SDK venv)
#   :8770  reachy_viewer.py  — dashboard              (system python)
#   :8772  reachy_chat.py    — voice-to-voice loop    (.venv, whisper)
#   :8773  reachy_memory.py  — face recognition       (SDK venv)

cd "$(dirname "$0")"

# Load .env and export everything in it (REACHY_URL, keys, …)
set -a; source .env 2>/dev/null; set +a
# reachy_camera.py takes a bare host, derive it from REACHY_URL
export REACHY_HOST=$(echo "${REACHY_URL:-http://192.168.12.240:8000}" | sed -E 's|https?://([^:/]+).*|\1|')

stop_all() {
  pkill -f "reachy_camera.py" 2>/dev/null
  pkill -f "reachy_viewer.py" 2>/dev/null
  pkill -f "reachy_chat.py"   2>/dev/null
  pkill -f "reachy_memory.py" 2>/dev/null
  pkill -f "reachy_vibeverse.py" 2>/dev/null
  pkill -f "reachy_telegram.py" 2>/dev/null
  pkill -f "reachy_bridge.py" 2>/dev/null
  echo "wonder stack stopped."
}

if [[ "$1" == "stop" ]]; then stop_all; exit 0; fi

echo "robot: $REACHY_URL"
if ! curl -s -m 4 -o /dev/null "$REACHY_URL/api/daemon/status"; then
  echo "⚠️  robot unreachable at $REACHY_URL — check WiFi / update REACHY_URL in .env"
  echo "    hint: arp -a | grep -i reachy"
  exit 1
fi

stop_all >/dev/null 2>&1
sleep 1

reachy_env/bin/python3 reachy_camera.py  > /tmp/reachy_camera.log 2>&1 &
python3                reachy_viewer.py  > /tmp/reachy_viewer.log 2>&1 &
.venv/bin/python3      reachy_chat.py    > /tmp/reachy_chat.log   2>&1 &
reachy_env/bin/python3 reachy_memory.py  > /tmp/reachy_memory.log 2>&1 &
python3                reachy_vibeverse.py > /tmp/vibeverse.log     2>&1 &
python3                reachy_telegram.py  > /tmp/telegram.log      2>&1 &
NO_WAKE=1 python3      reachy_bridge.py    > /tmp/reachy_bridge.log 2>&1 &

echo "starting… (camera takes ~10s to negotiate WebRTC)"
sleep 12

ok=0
for svc in "8771/status camera" "8770/perception viewer" "8772/state chat" "8773/current memory"; do
  port_path="${svc%% *}"; name="${svc##* }"
  if curl -s -m 3 "http://localhost:$port_path" > /dev/null; then
    echo "  ✅ $name  (:${port_path%%/*})"
    ok=$((ok+1))
  else
    echo "  ❌ $name  (:${port_path%%/*}) — see /tmp/reachy_${name}.log"
  fi
done

echo
[[ $ok -eq 4 ]] && echo "🤖 Wonder is up → http://localhost:8770" \
                || echo "partial start ($ok/4) — check logs above"
