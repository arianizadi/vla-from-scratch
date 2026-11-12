#!/usr/bin/env bash
# The graphical monitor of a training machine, in the browser of your own computer.
#   sim/pickup/scripts/monitor_remote.sh <ssh-host> <repo-directory-on-that-host> [port]
#
# Run it on your own computer, not on the training machine. It opens one ssh session that
#   - forwards localhost:<port> here to 127.0.0.1:<port> there (default 8765), and
#   - starts the monitor server there (python -m sim.pickup.monitor_server), listening on that
#     loopback address only. It uses the repository's .venv/bin/python if there is one and python3
#     otherwise: the server needs nothing but the standard library of Python 3.10 or newer.
# When the server answers through the forward, http://localhost:<port> is opened in your browser
# (with `open` on macOS, `xdg-open` elsewhere; without either the URL is only printed).
#
# Ctrl-C ends the ssh session, and the server ends with it. Nothing is left running on the training
# machine: the server is the foreground program of the session's terminal, so it is hung up when the
# session closes, and it also stops by itself when the session's process is gone
# (--exit-with-parent).
#
# <ssh-host> is whatever ssh accepts: a Host from ~/.ssh/config, or user@address. The directory is
# absolute, or relative to your home directory there (a leading "~/" is understood).
# The session never joins a shared ssh connection (ControlMaster): a forward and a server started
# through one would outlive this script. You may therefore be asked to authenticate again.
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <ssh-host> <repo-directory-on-that-host> [port]" >&2
  exit 2
fi
host=$1; directory=$2; port=${3:-8765}
case $port in ''|*[!0-9]*) echo "The port must be a number, not: $port" >&2; exit 2 ;; esac
directory=${directory#\~/}
url=http://localhost:$port

# Does an HTTP server answer on the local port? (ssh accepts connections on it before the server
# behind it is up, so connecting is not enough.)
answers() {
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$port/api/state"
  else
    ( exec 9<>"/dev/tcp/127.0.0.1/$port" && printf 'GET /api/state HTTP/1.0\r\n\r\n' >&9 &&
      read -r -t 2 line <&9 && [[ $line == HTTP/* ]] ) 2>/dev/null
  fi
}
if answers; then
  echo "Something already answers on local port $port. Choose another: $0 $host <directory> <port>" >&2
  exit 1
fi

# What runs on the training machine. Wrapped in `exec sh -c` so that it does not depend on the login
# shell there, and so that the server replaces every shell on the way: it is then the direct child
# of the ssh session.
quote() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }
# shellcheck disable=SC2016  # $1 and $2 are for the shell on the training machine
remote='cd -- "$1" || exit 1; python=python3; [ -x .venv/bin/python ] && python=.venv/bin/python; '
# shellcheck disable=SC2016
remote+='exec "$python" -m sim.pickup.monitor_server --port "$2" --exit-with-parent'

session=
close() {
  if [ -n "$session" ]; then
    kill "$session" 2>/dev/null || true
    wait "$session" 2>/dev/null || true
  fi
}
trap close EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# -tt: a terminal on the far side, so that closing the session hangs the server up, and Ctrl-C here
# reaches it. ssh runs in the background of this script (so the traps above fire at once) but keeps
# the terminal as its input.
exec 3<&0
ssh -tt -C -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -L "$port:127.0.0.1:$port" -- "$host" \
  "exec sh -c $(quote "$remote") sh $(quote "$directory") $port" <&3 &
session=$!

until answers; do
  if ! kill -0 "$session" 2>/dev/null; then
    session=
    echo "The ssh session ended before the monitor answered." >&2
    exit 1
  fi
  sleep 0.3
done

# The terminal is in ssh's raw mode from here on, hence the explicit carriage returns.
printf '\r\nMonitor: %s   (Ctrl-C ends the session and the server)\r\n' "$url"
if [ "$(uname)" = Darwin ] && command -v open >/dev/null 2>&1; then
  open "$url" || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 || true
else
  printf 'Open it in a browser.\r\n'
fi

wait "$session" || true
session=
echo "The session is closed and the monitor server has ended with it."
