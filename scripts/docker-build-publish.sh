#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_IMAGE="ghcr.io/nari-labs/nari-qwen3-tts"
readonly DEFAULT_PLATFORM="linux/amd64"
readonly GITHUB_HOST="github.com"
readonly GHCR_REGISTRY="ghcr.io"

image="${NARI_DOCKER_IMAGE:-$DEFAULT_IMAGE}"
platform="${NARI_DOCKER_PLATFORM:-$DEFAULT_PLATFORM}"
dry_run=false

usage() {
  cat <<'EOF'
Build and publish the Nari Qwen3-TTS image.

Usage:
  scripts/docker-build-publish.sh [options]

Options:
  --image IMAGE       Image repository (default: ghcr.io/nari-labs/nari-qwen3-tts)
  --platform PLATFORM Build platform (default: linux/amd64)
  --dry-run           Print the resolved tags and buildx command without building
  -h, --help          Show this help

Tag policy:
  Every build publishes sha-<full Git SHA> and latest.

Authentication:
  The script uses the active `gh auth` account to log Docker in to ghcr.io.
  The token must include write:packages. Add it with:
    gh auth refresh -h github.com -s write:packages
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || die "$option requires a value"
}

authenticate_ghcr() {
  local github_user
  local oauth_scopes

  command -v gh >/dev/null 2>&1 || die "gh is required"
  gh auth status --hostname "$GITHUB_HOST" >/dev/null 2>&1 ||
    die "gh is not authenticated for $GITHUB_HOST; run: gh auth login"

  oauth_scopes=$(
    gh api --hostname "$GITHUB_HOST" --include user 2>/dev/null |
      sed -n 's/^[Xx]-[Oo]auth-[Ss]copes: *//p' |
      tr -d '\r'
  )
  if [[ -n "$oauth_scopes" && ",${oauth_scopes// /}," != *",write:packages,"* ]]; then
    die "the active gh token is missing write:packages; run: gh auth refresh -h $GITHUB_HOST -s write:packages"
  fi

  github_user=$(gh api --hostname "$GITHUB_HOST" user --jq .login)
  printf 'Authenticating to %s as %s with gh...\n' "$GHCR_REGISTRY" "$github_user"
  gh auth token --hostname "$GITHUB_HOST" |
    docker login "$GHCR_REGISTRY" --username "$github_user" --password-stdin
}

while (($# > 0)); do
  case "$1" in
  --image)
    require_value "$1" "${2:-}"
    image="$2"
    shift 2
    ;;
  --platform)
    require_value "$1" "${2:-}"
    platform="$2"
    shift 2
    ;;
  --dry-run)
    dry_run=true
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    die "unknown option: $1"
    ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git is required"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir/.." rev-parse --show-toplevel)
commit=$(git -C "$repo_root" rev-parse HEAD)

[[ "$image" == "$GHCR_REGISTRY/"* ]] ||
  die "--image must use $GHCR_REGISTRY when authenticating with gh: $image"

if [[ "$dry_run" == false && -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  die "refusing to publish a dirty working tree; commit or stash changes first"
fi

tags=("sha-$commit" "latest")

build_command=(
  docker buildx build
  --platform "$platform"
  --build-arg "VCS_REF=$commit"
  --label "org.opencontainers.image.revision=$commit"
  --label "org.opencontainers.image.version=$commit"
  --provenance=mode=max
  --sbom=true
)

for tag in "${tags[@]}"; do
  build_command+=(--tag "$image:$tag")
done

build_command+=(--push "$repo_root")

printf 'Commit: %s\n' "$commit"
printf 'Image:  %s\n' "$image"
printf 'Tags:\n'
printf '  - %s\n' "${tags[@]}"

if [[ "$dry_run" == true ]]; then
  printf 'Command:\n  '
  printf '%q ' "${build_command[@]}"
  printf '\n'
  exit 0
fi

command -v docker >/dev/null 2>&1 || die "docker is required"
docker buildx version >/dev/null 2>&1 || die "docker buildx is required"
authenticate_ghcr

"${build_command[@]}"
