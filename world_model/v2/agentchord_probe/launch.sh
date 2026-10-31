#!/usr/bin/env bash
# agentchord_probe/launch.sh -- start the LLM gateway on the host, then run ac_run.py inside the EmbodiChain container.
#
# Usage: launch.sh RUN_NAME BACKEND GPU -- <ac_run.py args, without --out/--plan-dir/--gpu-id>
#   BACKEND: codex | qwen-jenga | replay | fake      GPU: westeros GPU index (must be free; caller holds the lock)
#   env PLAN_NAME (default RUN_NAME): which plan dir to use/regenerate (runs/plans/<PLAN_NAME>), so many runs can share one plan.
# Example: launch.sh e0_single_codex codex 6 -- --task SinglePourWater --episodes 20 --p 0.05 --recovery --regenerate
#
# Safety (AGENTS.md hardware rule): container runs --containall (no $HOME, no ~/.codex, no .env); only the clone (read-only),
# the probe code (read-only), the run dir and the asset cache are bound; no /dev beyond the GPU (--nv). ac_run.py blocks
# embodichain.deploy / ROS / CAN / robot-SDK imports. After the run the logs are grepped for hardware-path traces.
set -euo pipefail
RUN=$1; BACKEND=$2; GPU=$3; shift 3; [ "${1:-}" = "--" ] && shift
ROOT=/data/local/ss5772/agentchord
REPO=/common/users/ss5772/bimanual
PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
SIF=$ROOT/containers/embodichain.sif
OUT=$ROOT/runs/$RUN; PLAN=$ROOT/runs/plans/${PLAN_NAME:-$RUN}
PORT=${AC_PORT:-$((8790 + GPU))}
mkdir -p "$OUT/home" "$OUT/tmp" "$PLAN" "$ROOT/data"
# venv on top of the image's py310 (built 10-28): dexsim_engine 0.3.11 wheel (sha256 e3b7de08…, from DexForce's HTTP index, user OK
# 10-28) + their missing deps from PyPI; tensordict pinned 0.8.3 so the image's torch 2.7.0+cu128 stays. The image ships 0-byte NVIDIA
# driver placeholders in /usr/lib/x86_64-linux-gnu, so the host libs (/.singularity.d/libs, from --nv) must come first.
LDP="/.singularity.d/libs:/opt/lib/onnxruntime-linux-x64-gpu-1.17.3/lib:/opt/lib/tensorrt/TensorRT-10.11.0.33/lib:/usr/local/cuda/lib64/:/usr/lib/x86_64-linux-gnu"
[ -e "$OUT/episodes.jsonl" ] && { echo "refusing: $OUT/episodes.jsonl exists (pick a new RUN name)"; exit 1; }

case $BACKEND in
  # gpt-5.5 / gpt-5 are rejected for Codex on a ChatGPT account (10-27).  10-29 USER: the best GPT-5.6 (sol) for AgentChord, VIZ-COAST
  # and our layer for now (10-27..28 runs used terra); all three move to one open-weight model later.  The gateway logs the model per row.
  codex)      GW=(--backend codex --model "${AC_CODEX_MODEL:-gpt-5.6-sol}" --reasoning "${AC_CODEX_REASONING:-high}" --concurrency 1) ;;
  qwen-jenga) GW=(--backend openai --upstream "http://127.0.0.1:${AC_JENGA_PORT:-18781}/v1"
                  --upstream-model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --max-tokens 2048 --concurrency 1 --drop-images) ;;
  replay)     GW=(--backend replay) ;;
  fake)       GW=(--backend fake) ;;
  *) echo "unknown backend $BACKEND"; exit 1 ;;
esac
# qwen-jenga needs a tunnel the caller opened: ssh -N -L 18781:127.0.0.1:8781 ilab3   (jenga's rules: 1 in flight, <=2048 tokens)

cd "$REPO"
"$PY" -m world_model.v2.agentchord_probe.gateway "${GW[@]}" --port "$PORT" --log "$PLAN/llm_log.jsonl" \
  ${AC_REUSE:+--reuse} > "$OUT/gateway.log" 2>&1 &
GWPID=$!
trap 'kill $GWPID 2>/dev/null || true' EXIT
for _ in $(seq 50); do grep -q "gateway backend" "$OUT/gateway.log" 2>/dev/null && break; sleep 0.2; done

echo "{\"run\":\"$RUN\",\"backend\":\"$BACKEND\",\"gpu\":$GPU,\"port\":$PORT,\"plan\":\"$PLAN\",\"sif\":\"$SIF\",\"started\":\"$(date -Is)\"}" > "$OUT/launch.json"

# GPU isolation: DexSim renders through Vulkan, which IGNORES CUDA_VISIBLE_DEVICES (10-28: --nv + CUDA_VISIBLE_DEVICES=6 put
# every sim on physical GPU 0, jenga's). --nvccli + NVIDIA_VISIBLE_DEVICES exposes only /dev/nvidia$GPU inside the container.
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
watchdog() {  # kill this run if any of its processes shows up on a GPU other than $GPU
  while sleep 10; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r u p; do
      [ "$u" = "$UUID" ] && continue
      if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "/work/run --plan-dir" && \
         [ "$(stat -c %U /proc/$p 2>/dev/null)" = "$USER" ] && grep -q "$OUT" /proc/$p/environ 2>/dev/null; then
        echo "$(date -Is) GPU WATCHDOG: pid $p on $u (expected $UUID); killing" >> "$OUT/gpu_watchdog.txt"; kill -KILL "$p"
      fi
    done
  done
}
NVCAPS=compute,utility,graphics,display   # nvccli rejects "all"
NVIDIA_VISIBLE_DEVICES=$GPU NVIDIA_DRIVER_CAPABILITIES=$NVCAPS apptainer exec --containall --nvccli --writable-tmpfs \
  --env AC_OUT_HOST="$OUT" \
  -B "$ROOT/src:/work/src:ro" -B "$REPO/world_model/v2/agentchord_probe:/work/probe:ro" \
  -B "$OUT:/work/run" -B "$PLAN:/work/plan" -B "$ROOT/data:/work/data" \
  -B /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro \
  -B /usr/share/vulkan/implicit_layer.d/nvidia_layers.json:/usr/share/vulkan/implicit_layer.d/nvidia_layers.json:ro \
  -B /usr/share/nvidia:/usr/share/nvidia:ro \
  -B "$ROOT/venv:/work/venv:ro" -B "$OUT/tmp:/tmp" \
  --env HOME=/work/run/home --env EMBODICHAIN_DATA_ROOT=/work/data --env PYTHONPATH=/work/src \
  --env PYTHONNOUSERSITE=1 --env PYTHONFAULTHANDLER=1 --env TMPDIR=/work/run/tmp --env LD_LIBRARY_PATH="$LDP" \
  --env LLM_URL="http://127.0.0.1:$PORT/v1" --env OPENAI_API_KEY=gateway-local \
  --env NVIDIA_DRIVER_CAPABILITIES=$NVCAPS --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
  "$SIF" /work/venv/py310/bin/python /work/probe/ac_run.py --src /work/src --out /work/run --plan-dir /work/plan --gpu-id 0 "$@" \
  > "$OUT/run.log" 2>&1 &
CPID=$!
watchdog & WDPID=$!
trap 'kill $GWPID $WDPID 2>/dev/null || true' EXIT
wait $CPID || echo "ac_run exit $?" >> "$OUT/run.log"
kill $WDPID 2>/dev/null || true

# hardware-path audit (must be empty apart from our own block message)
if grep -nE "run_agent_realworld|rclpy|piper_sdk|can0|socketcan|ros2 " "$OUT/run.log" | grep -v "blocked (hardware path" > "$OUT/hw_audit.txt"; then
  echo "HW AUDIT HIT -- see $OUT/hw_audit.txt"; else echo "hw audit clean" > "$OUT/hw_audit.txt"; fi
echo "done: $OUT"
