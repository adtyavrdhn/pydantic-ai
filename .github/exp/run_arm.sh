#!/usr/bin/env bash
# One experiment cell. EXPMARK lines are parsed from the job log.
set -u
failed=0
# Set only in the arm that needs it: an empty UV_COMPILE_BYTECODE is an error to uv, and control arms must match real CI.
if [[ "$ARM" == compile ]]; then export UV_COMPILE_BYTECODE=1; fi
if [[ "$ARM" == ctrace ]]; then export COVERAGE_CORE=ctrace; fi
if [[ "$KIND" == durable ]]; then export TEMPORAL_DEBUG=1; fi
mark() { echo "EXPMARK $1 $(date +%s)"; }
SHARD=1
if [[ "$ARM" == third* ]]; then SHARD="${ARM#third}"; fi
# Every arm but control installs the candidate lockfile, which differs from upstream's only in coverage (and tomli).
if [[ "$ARM" == cov717 || "$ARM" == third* ]]; then cp "$RUNNER_TEMP/exp/uv.lock.717" uv.lock; fi
EXTRAS=(--all-packages --all-extras --no-extra mcp-tasks)
cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'); echo "EXPCPU $cpu nproc=$(nproc)"
narrow_modules() { git grep -liE 'temporal' -- 'tests/harness/**/test_*.py' 'tests/harness/test_*.py' | sort -u; }

if [[ "$KIND" == verify ]]; then
  uv sync "${EXTRAS[@]}" || exit 1
  uv run --no-sync pytest --collect-only -q -m temporal tests/harness 2>/dev/null | grep '::' | sort > /tmp/full.txt
  mapfile -t mods < <(narrow_modules)
  uv run --no-sync pytest --collect-only -q -m temporal "${mods[@]}" 2>/dev/null | grep '::' | sort > /tmp/narrow.txt
  echo "EXPVERIFY full=$(wc -l < /tmp/full.txt) narrow=$(wc -l < /tmp/narrow.txt) modules=${#mods[@]}"
  diff /tmp/full.txt /tmp/narrow.txt && echo "EXPVERIFY identical" || { echo "EXPVERIFY DIFFERENT"; exit 1; }
  exit 0
fi

if [[ "$KIND" == nocuda ]]; then
  no_install=()
  for p in $(grep -E '^name = "(nvidia-.*|cuda-.*|triton)"' uv.lock | sed -E 's/name = "(.*)"/\1/'); do no_install+=(--no-install-package "$p"); done
  echo "EXPNOCUDA excluded ${no_install[*]}"
  mark sync_start
  uv sync "${EXTRAS[@]}" "${no_install[@]}" || { echo "EXPNOCUDA sync failed"; exit 1; }
  mark sync_end
  du -sh .venv; du -sh "${UV_CACHE_DIR:-$HOME/.cache/uv}" 2>/dev/null
  uv run --no-sync python -c "
import torch; print('EXPNOCUDA torch', torch.__version__, 'cuda', torch.cuda.is_available())
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers-testing/stsb-bert-tiny-safetensors', revision='f3cb857cba53019a20df283396bcca179cf051a4')
print('EXPNOCUDA encoded', m.encode(['hello']).shape)
" || echo "EXPNOCUDA import failed"
  uv run --no-sync pytest tests/test_embeddings.py -q -k "sentence or Sentence" -p no:cacheprovider 2>&1 | tail -5
  mark tests_end
  exit 0
fi

export COVERAGE_PROCESS_START="$PWD/pyproject.toml"
runner=(coverage run -m pytest)
if [[ "$ARM" == nocov ]]; then
  # What a job without coverage runs today: see `COLLECT_COVERAGE` in ci.yml.
  unset COVERAGE_PROCESS_START
  runner=(pytest)
fi
if [[ "$ARM" == include ]]; then
  # Same settings as pyproject.toml, tracing only the files only 3.11 can cover.
  uv run "${EXTRAS[@]}" python "$RUNNER_TEMP/exp/make_rc.py" "$RUNNER_TEMP/exp/keep311.txt" "$RUNNER_TEMP/exp/covrc"
  export COVERAGE_PROCESS_START="$RUNNER_TEMP/exp/covrc"
  runner=(coverage run --rcfile="$RUNNER_TEMP/exp/covrc" -m pytest)
fi
mark sync_start
uv run "${EXTRAS[@]}" python -c "print('synced')" || exit 1
mark sync_end

if [[ "$KIND" == durable ]]; then
  mark p1_start
  uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup tests/durable_exec || { echo "EXPFAIL p1 $?"; failed=1; }
  mark p1_end
  paths=(tests/harness)
  if [[ "$ARM" == narrow ]]; then mapfile -t paths < <(narrow_modules); fi
  mark p2_start
  uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup -m temporal "${paths[@]}" || { echo "EXPFAIL p2 $?"; failed=1; }
  mark p2_end
  exit $failed
fi

# all-extras, shard 1 of the real split, chosen from a fixed baseline so every arm runs the same modules.
if [[ "$PY" == 3.11 ]]; then
  uv run "${EXTRAS[@]}" python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers-testing/stsb-bert-tiny-safetensors', revision='f3cb857cba53019a20df283396bcca179cf051a4')" || true
fi
python3 .github/scripts/shard_durations.py select --durations "$RUNNER_TEMP/exp/$PY-all-extras.json" --shards "$SHARDS" --shard "$SHARD" --exclude tests/durable_exec > /tmp/shard-args.txt
mapfile -t shard_args < /tmp/shard-args.txt
if [[ "$ARM" == warm ]]; then
  mark warm_start
  env -u COVERAGE_PROCESS_START uv run "${EXTRAS[@]}" pytest --collect-only -q -p no:cacheprovider --ignore=tests/durable_exec -m "not temporal" "${shard_args[@]}" > /dev/null || true
  mark warm_end
fi
mark pytest_start
uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup --ignore=tests/durable_exec -m "not temporal" "${shard_args[@]}" || { echo "EXPFAIL pytest $?"; failed=1; }
mark pytest_end
uv run --no-sync coverage --version | head -1 | sed "s/^/EXPCOVVERSION /"
if [[ "$ARM" == include ]]; then
  # Prove the arm traced what it claims: count the measured files after combining.
  uv run "${EXTRAS[@]}" coverage combine --rcfile="$RUNNER_TEMP/exp/covrc" -q --data-file="$COVERAGE_FILE" || true
  python3 -c "import os, sqlite3; c = sqlite3.connect(os.environ['COVERAGE_FILE']); print('EXPINCLUDE files', c.execute('select count(*) from file').fetchone()[0], 'arcs', c.execute('select count(*) from arc').fetchone()[0])" || echo "EXPINCLUDE unreadable"
fi
exit $failed
