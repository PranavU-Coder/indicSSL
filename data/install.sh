#!/usr/bin/env bash
# Usage:
#   bash download_kathbath.sh                         # clean split -> ./data
#   bash download_kathbath.sh --dest ~/Projects/indicJEPA/data, you might wanna change accordingly
#   bash download_kathbath.sh --no-train              # skip the 85 GB train tar (eval-only)
#   bash download_kathbath.sh --split noisy           # noisy test sets + transcripts
#   bash download_kathbath.sh --split both
#   bash download_kathbath.sh --convert --superb-dir ~/Projects/indicJEPA/IndicSUPERB
#                                                     # also m4a -> 16 kHz wav
#
# resumable (wget -c). re-running skips finished downloads && finished extracts.

set -euo pipefail

DEST="./data"
SPLIT="clean"          
GET_TRAIN=1
CONVERT=0
INDICSUPERB_DIR="${INDICSUPERB_DIR:-}"
BASE="https://objectstore.e2enetworks.net/indic-superb/kathbath"
LANGS=(bengali gujarati hindi kannada malayalam marathi odia punjabi sanskrit tamil telugu urdu)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)       DEST="$2"; shift 2;;
    --split)      SPLIT="$2"; shift 2;;
    --no-train)   GET_TRAIN=0; shift;;
    --convert)    CONVERT=1; shift;;
    --superb-dir) INDICSUPERB_DIR="$2"; shift 2;;
    -h|--help)    sed -n '2,27p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# guys please install wget, it's 2026
command -v wget >/dev/null || { echo "wget not found"; exit 1; }

TARS_ROOT="$DEST/tars"
mkdir -p "$TARS_ROOT"

process_split() {
  local split="$1"
  local tdir="$TARS_ROOT/$split"
  local m4a
  [[ "$split" == clean ]] && m4a="$DEST/kb_data_clean_m4a" || m4a="$DEST/kb_data_noisy_m4a"
  mkdir -p "$tdir" "$m4a"

  local list=()
  if [[ "$split" == clean ]]; then
    [[ $GET_TRAIN == 1 ]] && list+=("$BASE/clean/train_audio.tar")
    list+=("$BASE/clean/valid_audio.tar" "$BASE/clean/testkn_audio.tar" \
           "$BASE/clean/testunk_audio.tar" "$BASE/clean/transcripts_n2w.tar")
  else
    list+=("$BASE/noisy/testkn_audio.tar" "$BASE/noisy/testunk_audio.tar" \
           "$BASE/noisy/transcripts_n2w.tar")
  fi

  echo "== split: $split  (${#list[@]} files -> $m4a) =="
  for u in "${list[@]}"; do
    local fn base sentinel
    fn="$tdir/$(basename "$u")"
    base="$(basename "$u" .tar)"
    sentinel="$tdir/.extracted_${base}"
    echo ">> $(basename "$u")"
    wget -c -P "$tdir" "$u"
    if [[ -f "$sentinel" ]]; then echo "   already extracted, skip"; continue; fi
    echo "   verifying archive…"
    tar -tf "$fn" >/dev/null 2>&1 || { echo "!! integrity fail: $fn (delete it and re-run to resume)"; exit 1; }
    echo "   extracting -> $m4a"
    tar -xf "$fn" -C "$m4a"
    touch "$sentinel"
  done

  if [[ -d "$m4a/data" ]]; then
    echo "   flattening $m4a/data/* up one level"
    shopt -s dotglob
    mv "$m4a"/data/* "$m4a"/ 2>/dev/null || true
    rmdir "$m4a/data" 2>/dev/null || true
    shopt -u dotglob
  fi

  echo "   layout (dirs, depth 2):"
  find "$m4a" -maxdepth 2 -type d | sort | sed 's/^/     /' | head -40
}

case "$SPLIT" in
  clean) process_split clean;;
  noisy) process_split noisy;;
  both)  process_split clean; process_split noisy;;
  *) echo "bad --split: $SPLIT (clean|noisy|both)"; exit 2;;
esac

if [[ $CONVERT == 1 ]]; then
  echo
  local_m4a="$DEST/kb_data_clean_m4a"
  wav="$DEST/kb_data_clean_wav"
  command -v ffmpeg >/dev/null || { echo "ffmpeg not found (pacman -S ffmpeg)"; exit 1; }
  if [[ -z "$INDICSUPERB_DIR" || ! -f "$INDICSUPERB_DIR/utilities/structure.py" ]]; then
    echo "IndicSUPERB repo not found. Re-run with:"
    echo "  --convert --superb-dir /path/to/IndicSUPERB"
    echo "Per language it runs:"
    echo "  python \$SUPERB/utilities/structure.py $local_m4a $wav <lang>"
    exit 1
  fi
  mkdir -p "$wav"
  for l in "${LANGS[@]}"; do
    [[ -d "$local_m4a/$l" ]] || { echo "skip $l (not present)"; continue; }
    echo ">> converting $l -> wav"
    python "$INDICSUPERB_DIR/utilities/structure.py" "$local_m4a" "$wav" "$l"
  done
  echo
  echo "wav data ready: $wav"
  echo "point downstream at:  config.downstream_expert.datarc.file_path=$wav"
fi

echo
echo "done."