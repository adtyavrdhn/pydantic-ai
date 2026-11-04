#!/usr/bin/env bash
# One experiment cell. EXPMARK lines are parsed from the job log.
set -u
mark() { echo "EXPMARK $1 $(date +%s)"; }
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
  du -sh .venv; du -sh "$UV_CACHE_DIR" 2>/dev/null
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
mark sync_start
uv run "${EXTRAS[@]}" python -c "print('synced')" || exit 1
mark sync_end

if [[ "$KIND" == durable ]]; then
  mark p1_start
  uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup tests/durable_exec || echo "EXPFAIL p1 $?"
  mark p1_end
  paths=(tests/harness)
  if [[ "$ARM" == narrow ]]; then mapfile -t paths < <(narrow_modules); fi
  mark p2_start
  uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup -m temporal "${paths[@]}" || echo "EXPFAIL p2 $?"
  mark p2_end
  exit 0
fi

# all-extras, shard 1 of the real split, chosen from a fixed baseline so every arm runs the same modules.
if [[ "$PY" == 3.11 ]]; then
  uv run "${EXTRAS[@]}" python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers-testing/stsb-bert-tiny-safetensors', revision='f3cb857cba53019a20df283396bcca179cf051a4')" || true
fi
python3 .github/scripts/shard_durations.py select --durations ".github/exp/$PY-all-extras.json" --shards "$SHARDS" --shard 1 --exclude tests/durable_exec > /tmp/shard-args.txt
mapfile -t shard_args < /tmp/shard-args.txt
if [[ "$ARM" == warm ]]; then
  mark warm_start
  env -u COVERAGE_PROCESS_START uv run "${EXTRAS[@]}" pytest --collect-only -q -p no:cacheprovider --ignore=tests/durable_exec -m "not temporal" "${shard_args[@]}" > /dev/null || true
  mark warm_end
fi
mark pytest_start
uv run "${EXTRAS[@]}" "${runner[@]}" --durations=5 -n logical --dist=loadgroup --ignore=tests/durable_exec -m "not temporal" "${shard_args[@]}" || echo "EXPFAIL pytest $?"
mark pytest_end
