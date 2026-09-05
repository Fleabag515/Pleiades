#!/bin/bash
# Phase 4 bench harness: wall-clock comparison of the engines that can serve a
# GGUF on this machine, identical model + placement. Reports cold prefill
# (long prompt, 1 output token) and decode (tiny prompt, N output tokens) --
# the two numbers that decide perceived latency for a single-user workstation.
# Optional engines are skipped when not installed (ollama needs the same model
# pulled under the same name).
#
# Usage: compare.sh /path/to/model.gguf [port] [ollama-model-name]
set -u
MODEL=${1:?usage: compare.sh model.gguf [port] [ollama-model]}
NGL=${NGL:-999}
PORT=${2:-8799}
OLLAMA_MODEL=${3:-}
PROMPT=$(python3 -c "import json;print(json.dumps([{'role':'user','content':'Here is a long story. '*190+' Now summarize it in one word.'}]))")
REPLY() { python3 -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'][:30])" 2>/dev/null || echo "(parse failed)"; }

bench() { # $1=label $2=boot-cmd-array-eval $3=health-wait-seconds
  echo "=== $1"
  local PID
  eval "$2" >/tmp/bench-$1.log 2>&1 &
  PID=$!
  for _ in $(seq 1 "$3"); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break; sleep 0.3; done
  curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "boot FAILED"; kill $PID 2>/dev/null; return 1; }
  printf "  cold prefill (%s tok, 1 out):  " 1177
  curl -s -o /dev/null -w "%{time_total}s\n" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"messages\":$PROMPT,\"max_tokens\":1,\"temperature\":0}"
  printf "  decode (12 tok, 128 out):      "
  curl -s -o /dev/null -w "%{time_total}s\n" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":128,"temperature":0}'
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
}

ENGINE=~/Pleiades/engine/build/pleiades-engine-server
LS=$(ls ~/.pleiades/runtime/cuda-main/build/bin/llama-server 2>/dev/null)
[ -x "$LS" ] || LS=$(find ~/.pleiades/runtime -name llama-server -type f 2>/dev/null | head -1)

bench engine "$ENGINE --model $MODEL --host 127.0.0.1 --port $PORT --ngl $NGL --fa on --ctx 2048 --alias bench" 150
if [ -n "$LS" ] && [ -x "$LS" ]; then
  export LD_LIBRARY_PATH="$(dirname "$LS"):$LD_LIBRARY_PATH"
  bench llama-server "$LS -m $MODEL --host 127.0.0.1 --port $PORT -ngl $NGL -fa on -c 2048 --alias bench" 200
else
  echo "=== llama-server: not installed, skipped"
fi
if [ -n "$OLLAMA_MODEL" ] && ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
  bench ollama "[ollama serve]" 30
  # ollama's OpenAI endpoint lives on :11434 and needs the model pre-pulled;
  # measured separately by the caller if desired.
  echo "  (ollama endpoint: http://127.0.0.1:11434/v1 -- model $OLLAMA_MODEL)"
else
  echo "=== ollama: server or model not present, skipped"
fi
echo BENCH-DONE
