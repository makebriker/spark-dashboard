#!/usr/bin/env bash
set -euo pipefail

GB10_HOST="${GB10_HOST:-}"
GB10_USER="${GB10_USER:-$USER}"
REMOTE_DIR="/home/${GB10_USER}/spark-dashboard"

if [ -z "$GB10_HOST" ]; then
  echo "Usage: GB10_HOST=<ip> ./deploy.sh"
  echo "  Optionally set GB10_USER (default: \$USER)"
  exit 1
fi

echo "==> Deploying spark-dashboard to ${GB10_USER}@${GB10_HOST}:${REMOTE_DIR}"

echo "==> Syncing files..."
rsync -avz --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  "$(dirname "$0")/" "${GB10_USER}@${GB10_HOST}:${REMOTE_DIR}/"

echo "==> Setting up venv and installing dependencies..."
ssh "${GB10_USER}@${GB10_HOST}" bash -s << 'REMOTE'
set -euo pipefail
cd ~/spark-dashboard

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -q -r requirements.txt

mkdir -p ~/.config/systemd/user
cp systemd/spark-dashboard.service ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now spark-dashboard 2>/dev/null || true

loginctl enable-linger "$(whoami)" 2>/dev/null || true

echo "==> Dashboard status:"
systemctl --user status spark-dashboard --no-pager || true

echo ""
echo "==> Dashboard available at: http://$(hostname -I | awk '{print $1}'):8500"
REMOTE

echo "==> Done!"