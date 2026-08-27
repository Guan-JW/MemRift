#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tag=${TAG:-memrift-artifact:0.1.0-review}
version=${VERSION:-0.1.0-review}
doi=${DOI:?DOI must identify the archival release}
include_image=${INCLUDE_IMAGE:-1}
output_dir=${RELEASE_DIR:-"$root/dist"}
results_dir=${RESULTS_DIR:-"$root/results"}
dataset_receipt=${DATASET_RECEIPT:-"$root/.cache/huggingface/memrift-dataset-receipt.json"}
training_checkpoint_dir=${CHECKPOINT_DIR:?CHECKPOINT_DIR must identify the Zstd-18 training checkpoint}
loading_checkpoint_dir=${LOADING_CHECKPOINT_DIR:?LOADING_CHECKPOINT_DIR must identify the loading checkpoints}
tables23_manifest="$results_dir/tables23-tinyllama-1.1b-chat-v1.0/tables23_manifest.json"
archive="$output_dir/memrift-artifact-$version-release.tar.zst"

case "$include_image" in
  0|1) ;;
  *) printf 'INCLUDE_IMAGE must be 0 or 1\n' >&2; exit 2 ;;
esac
commands=(git sha256sum tar zstd)
test "$include_image" = 0 || commands+=(docker)
for command in "${commands[@]}"; do
  command -v "$command" >/dev/null || { printf '%s is required\n' "$command" >&2; exit 2; }
done
for path in "$results_dir" "$dataset_receipt" "$training_checkpoint_dir/index.json" \
  "$training_checkpoint_dir/metadata.json" "$loading_checkpoint_dir/nf4/preparation.json" \
  "$loading_checkpoint_dir/memrift/preparation.json" "$tables23_manifest"; do
  test -e "$path" || { printf 'required release input is missing: %s\n' "$path" >&2; exit 2; }
done
test -z "$(git -C "$root" status --porcelain)" || {
  printf 'release packaging requires a clean source tree\n' >&2
  exit 2
}
if test -e "$archive" && test "${FORCE:-0}" != 1; then
  printf 'refusing to overwrite %s; set FORCE=1 to replace it\n' "$archive" >&2
  exit 2
fi

image_id=
if test "$include_image" = 1; then
  image_id=$(docker image inspect "$tag" --format '{{.Id}}')
fi
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$output_dir"
stage=$(mktemp -d "${TMPDIR:-/tmp}/memrift-release.XXXXXX")
trap 'rm -rf "$stage"' EXIT
bundle="$stage/memrift-artifact-$version"
mkdir -p "$bundle/provenance/training-checkpoint" "$bundle/provenance/loading-nf4" \
  "$bundle/provenance/loading-memrift" "$bundle/provenance/dataset"
source_revision=$(git -C "$root" rev-parse HEAD)
printf '%s\n' "$source_revision" > "$stage/source-revision.txt"

if test "$include_image" = 1; then
  docker save "$tag" | zstd -T2 -10 -q -o "$bundle/image.tar.zst"
fi

source_tar="$stage/source.tar"
git -C "$root" ls-files -z --cached --others --exclude-standard | \
  tar -C "$root" --null --files-from=- --sort=name --mtime=@0 \
    --owner=0 --group=0 --numeric-owner -cf "$source_tar"
tar -C "$stage" --append -f "$source_tar" \
  --mtime=@0 --owner=0 --group=0 --numeric-owner \
  --transform='s|^source-revision.txt$|.memrift-source-revision|' source-revision.txt
zstd -T2 -10 -q "$source_tar" -o "$bundle/source.tar.zst"
rm -f "$source_tar"

tar -C "$(dirname "$results_dir")" --sort=name --mtime=@0 \
  --owner=0 --group=0 --numeric-owner -cf - "$(basename "$results_dir")" | \
  zstd -T2 -10 -q -o "$bundle/evidence.tar.zst"

install -m 0644 "$training_checkpoint_dir/index.json" "$bundle/provenance/training-checkpoint/index.json"
install -m 0644 "$training_checkpoint_dir/metadata.json" "$bundle/provenance/training-checkpoint/metadata.json"
install -m 0644 "$loading_checkpoint_dir/nf4/preparation.json" "$bundle/provenance/loading-nf4/preparation.json"
install -m 0644 "$loading_checkpoint_dir/memrift/preparation.json" "$bundle/provenance/loading-memrift/preparation.json"
install -m 0644 "$dataset_receipt" "$bundle/provenance/dataset/memrift-dataset-receipt.json"
tar -C "$bundle" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -cf - provenance | zstd -T2 -10 -q -o "$bundle/provenance.tar.zst"
rm -rf "$bundle/provenance"

python3 - "$tables23_manifest" "$bundle/RELEASE.json" "$version" "$tag" "$image_id" "$created_at" "$source_revision" "$doi" "$include_image" <<'PY'
import json
import sys
from pathlib import Path

evidence = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
include_image = sys.argv[9] == "1"
image = {"registry_reference": sys.argv[4], "redistributed": include_image}
if include_image:
    image.update({"id": sys.argv[5], "archive": "image.tar.zst"})
notes = [
    "Generated results are preserved verbatim in evidence.tar.zst.",
    "Model weights and checkpoints are not redistributed; provenance receipts and checkpoint metadata are included.",
]
if not include_image:
    notes.append("The container image is referenced by immutable registry digest and is not redistributed in this archive.")
release = {
    "schema_version": "1.0",
    "version": sys.argv[3],
    "doi": sys.argv[8],
    "archive_url": f"https://doi.org/{sys.argv[8]}",
    "created_at": sys.argv[6],
    "source_revision": sys.argv[7],
    "image": image,
    "source_archive": "source.tar.zst",
    "evidence_archive": "evidence.tar.zst",
    "provenance_archive": "provenance.tar.zst",
    "tables23_status": evidence["status"],
    "tables23_protocol_reportable": evidence["protocol_reportable"],
    "notes": notes,
}
Path(sys.argv[2]).write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
  cd "$bundle"
checksum_files=(RELEASE.json evidence.tar.zst provenance.tar.zst source.tar.zst)
test "$include_image" = 0 || checksum_files+=(image.tar.zst)
sha256sum "${checksum_files[@]}" > SHA256SUMS
)
rm -f "$archive" "$archive.sha256"
tar -C "$stage" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -cf - "$(basename "$bundle")" | zstd -T2 -1 -q -o "$archive"
(
  cd "$output_dir"
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)
printf '%s\n' "$archive" "$archive.sha256"
