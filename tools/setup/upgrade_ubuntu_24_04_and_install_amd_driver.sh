#!/usr/bin/env bash
set -euo pipefail

# GRETA CORE – Ubuntu 24.04 upgrade + AMD official driver install
# Usage:
#   sudo GRETA_CONFIRM=1 \
#     DRIVER_URL="https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb" \
#     DRIVER_SHA256="<optional sha256>" \
#     ./tools/setup/upgrade_ubuntu_24_04_and_install_amd_driver.sh
#
# Notes:
# - This performs an in-place upgrade to Ubuntu 24.04.
# - It then installs AMD's official Radeon/Ryzen AI driver package from DRIVER_URL.
# - Reboots are required during the process.

log() { echo "[GRETA-UPGRADE] $*"; }

require_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "ERROR: must be run as root (use sudo)." >&2
    exit 1
  fi
}

require_confirm() {
  if [[ "${GRETA_CONFIRM:-}" != "1" ]]; then
    echo "ERROR: set GRETA_CONFIRM=1 to proceed (in-place upgrade + driver install)." >&2
    exit 1
  fi
}

require_ubuntu() {
  if [[ ! -f /etc/os-release ]]; then
    echo "ERROR: /etc/os-release not found." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: This script supports Ubuntu only. Detected: ${ID:-unknown}." >&2
    exit 1
  fi
}

preflight_report() {
  # shellcheck disable=SC1091
  . /etc/os-release
  log "Detected Ubuntu ${VERSION_ID:-unknown} (${VERSION_CODENAME:-unknown})."
  log "Kernel: $(uname -r)"
  log "RAM: $(free -h | awk '/Mem:/ {print $2}')"
}

upgrade_to_24_04() {
  log "Updating packages on current release..."
  apt-get update -y
  apt-get dist-upgrade -y

  log "Ensuring update-manager-core is installed..."
  apt-get install -y update-manager-core

  log "Setting release upgrader prompt to normal..."
  sed -i 's/^Prompt=.*/Prompt=normal/' /etc/update-manager/release-upgrades

  log "Starting in-place upgrade to Ubuntu 24.04 (do-release-upgrade)..."
  # Non-interactive upgrade is brittle; we run in text mode and let Ubuntu prompt if needed.
  do-release-upgrade -f DistUpgradeViewText

  log "Upgrade finished. A reboot is required."
}

install_amd_driver() {
  local url="${DRIVER_URL:-https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb}"

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  log "Downloading AMD driver package..."
  local deb_path="${tmpdir}/amdgpu-install.deb"
  curl -L "${url}" -o "${deb_path}"

  if [[ -n "${DRIVER_SHA256:-}" ]]; then
    log "Verifying SHA256..."
    echo "${DRIVER_SHA256}  ${deb_path}" | sha256sum -c -
  fi

  log "Installing driver package..."
  apt-get install -y "${deb_path}"

  log "Running amdgpu-install..."
  # Typical options for ROCm/Ryzen AI stack; adjust if AMD docs require different flags.
  amdgpu-install -y --usecase=rocm

  log "Driver install complete. Reboot required."
}

main() {
  require_root
  require_confirm
  require_ubuntu
  preflight_report

  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${VERSION_ID:-}" != "24.04" ]]; then
    upgrade_to_24_04
    log "Please reboot now, then re-run this script to install the driver."
    exit 0
  fi

  install_amd_driver
  log "All done. Please reboot."
}

main "$@"
