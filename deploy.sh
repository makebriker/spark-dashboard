#!/usr/bin/env bash
#
# Deploy spark-dashboard to a GB10 host.
#
#   HOST=192.0.2.10 ./deploy.sh
#
# The remote checkout at ~/spark-dashboard is a git clone of this repository, so
# deploying is a fetch + hard reset. On first run against a host whose directory
# is not yet a git repo, the script tarballs the existing contents and converts
# the directory in place, preserving the gitignored .venv/.
set -euo pipefail

HOST="${HOST:-}"
USER_REMOTE="${USER_REMOTE:-lesha}"
BRANCH="${BRANCH:-main}"
REPO="${REPO:-https://github.com/makebriker/spark-dashboard.git}"

if [ -z "$HOST" ]; then
  echo "Usage: HOST=<ip-or-hostname> ./deploy.sh" >&2
  echo "  Optional: USER_REMOTE (default: lesha), BRANCH (default: main)" >&2
  exit 1
fi

echo "==> Deploying $BRANCH to ${USER_REMOTE}@${HOST}"

ssh "${USER_REMOTE}@${HOST}" REPO="$REPO" BRANCH="$BRANCH" bash -s <<'REMOTE'
set -euo pipefail
cd "$HOME/spark-dashboard"

if [ ! -d .git ]; then
  stamp=$(date +%Y%m%d-%H%M%S)
  backup="$HOME/spark-dashboard-backup-${stamp}.tgz"
  echo "==> Not a git checkout yet; backing up to $backup"
  tar czf "$backup" --exclude=.venv --exclude=__pycache__ -C "$HOME" spark-dashboard

  # Untracked leftovers that would survive the checkout and show as dirty.
  mkdir -p "$HOME/spark-dashboard-preserved-${stamp}"
  find spark_dashboard -maxdepth 1 -name '*.pre-*' -exec mv {} "$HOME/spark-dashboard-preserved-${stamp}/" \; 2>/dev/null || true

  echo "==> Converting to a git checkout of $REPO"
  git init -q
  git remote add origin "$REPO"
fi

git fetch -q origin "$BRANCH"
git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
git reset -q --hard "origin/$BRANCH"
echo "==> At $(git log --oneline -1)"

if [ ! -d .venv ]; then
  echo "==> Creating venv"
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

mkdir -p "$HOME/.config/systemd/user"
cp systemd/spark-dashboard.service "$HOME/.config/systemd/user/"

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user daemon-reload
systemctl --user enable spark-dashboard >/dev/null 2>&1 || true
systemctl --user restart spark-dashboard
loginctl enable-linger "$(whoami)" >/dev/null 2>&1 || true

# nvitop/NVML init can take a few seconds; poll rather than check once.
ok=0
for _ in $(seq 1 15); do
  if curl -sf -o /dev/null http://localhost:8500/api/metrics; then ok=1; break; fi
  sleep 1
done
if [ "$ok" = 1 ]; then
  echo "==> OK: dashboard responding on :8500"
else
  echo "==> FAILED: /api/metrics not responding" >&2
  systemctl --user status spark-dashboard --no-pager || true
  exit 1
fi

echo "==> Working tree:"
git status --short || true
REMOTE

echo "==> Done: ${HOST}"
