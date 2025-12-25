#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

log() {
  printf "\n[%s] %s\n" "beatsmith" "$1"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

need_sudo() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "sudo"
  fi
}

ensure_python() {
  if ! has_cmd python3; then
    log "python3 not found. Attempting to install via system package manager."
    install_packages python3
  fi

  local py_version
  py_version="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  local major="${py_version%%.*}"
  local minor="${py_version##*.}"

  if [ "$major" -lt 3 ] || [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; then
    log "Python ${py_version} detected. BeatSmith requires Python 3.10+."
    log "Please install a newer Python (e.g., via pyenv or your OS package manager), then re-run this script."
    exit 1
  fi
}

install_packages() {
  local packages=("$@")
  local sudo_cmd
  sudo_cmd="$(need_sudo)"

  if has_cmd apt-get; then
    log "Installing dependencies with apt-get: ${packages[*]}"
    ${sudo_cmd} apt-get update -y
    ${sudo_cmd} apt-get install -y "${packages[@]}"
  elif has_cmd dnf; then
    log "Installing dependencies with dnf: ${packages[*]}"
    ${sudo_cmd} dnf install -y "${packages[@]}"
  elif has_cmd yum; then
    log "Installing dependencies with yum: ${packages[*]}"
    ${sudo_cmd} yum install -y "${packages[@]}"
  elif has_cmd pacman; then
    log "Installing dependencies with pacman: ${packages[*]}"
    ${sudo_cmd} pacman -Sy --noconfirm "${packages[@]}"
  elif has_cmd zypper; then
    log "Installing dependencies with zypper: ${packages[*]}"
    ${sudo_cmd} zypper --non-interactive install "${packages[@]}"
  elif has_cmd apk; then
    log "Installing dependencies with apk: ${packages[*]}"
    ${sudo_cmd} apk add "${packages[@]}"
  elif has_cmd brew; then
    log "Installing dependencies with brew: ${packages[*]}"
    brew install "${packages[@]}"
  else
    log "No supported package manager found. Please install: ${packages[*]}"
    exit 1
  fi
}

ensure_ffmpeg() {
  if has_cmd ffmpeg; then
    return
  fi

  log "ffmpeg not found. Attempting to install."

  if has_cmd apt-get; then
    install_packages ffmpeg
  elif has_cmd dnf; then
    install_packages ffmpeg
  elif has_cmd yum; then
    install_packages ffmpeg
  elif has_cmd pacman; then
    install_packages ffmpeg
  elif has_cmd zypper; then
    install_packages ffmpeg
  elif has_cmd apk; then
    install_packages ffmpeg
  elif has_cmd brew; then
    install_packages ffmpeg
  else
    log "Unable to install ffmpeg automatically. Please install ffmpeg and re-run this script."
    exit 1
  fi
}

ensure_python_deps() {
  if has_cmd apt-get; then
    install_packages python3 python3-venv python3-pip
  elif has_cmd dnf; then
    install_packages python3 python3-pip
  elif has_cmd yum; then
    install_packages python3 python3-pip
  elif has_cmd pacman; then
    install_packages python python-pip
  elif has_cmd zypper; then
    install_packages python3 python3-pip
  elif has_cmd apk; then
    install_packages python3 py3-pip
  elif has_cmd brew; then
    install_packages python
  fi
}

create_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at ${VENV_DIR}".
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${ROOT_DIR}"
}

log "Starting BeatSmith installer."

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! has_cmd brew; then
    log "Homebrew is required on macOS for dependency installation."
    log "Install it from https://brew.sh/ then re-run this script."
    exit 1
  fi
fi

ensure_python_deps
ensure_python
ensure_ffmpeg
create_venv

log "Install complete. Activate with: source .venv/bin/activate"
