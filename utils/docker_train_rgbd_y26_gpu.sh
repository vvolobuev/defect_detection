#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/defects_detection_project}"
CONTAINER="${CONTAINER:-yolo_rgbd4ch_y26n640_gpu}"
IMAGE="${IMAGE:-ultralytics/ultralytics:latest}"
FRACTION="${FRACTION:-1.0}"
WORKERS="${WORKERS:-8}"
NAME="${NAME:-defects_rgbd_y26n640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"

DOCKER_GPU_ARGS=(--gpus all)
DOCKER_OMP_TRAIN=(-e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}")
if [[ "${DEVICE,,}" == "cpu" ]]; then
  DOCKER_GPU_ARGS=()
  DOCKER_OMP_TRAIN=(-e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}")
fi

NO_AMP_ARGS=()
if [[ "${DEVICE,,}" == "cpu" ]] || [[ "${NO_AMP:-0}" == "1" ]]; then
  NO_AMP_ARGS=(--no-amp)
fi

mkdir -p "$PROJECT/yolo_runs"

docker pull "$IMAGE"

echo "=== Ultralytics Docker: VERIFY 4 канала через train_rgbd_4ch_yolo.py ==="
docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace/project \
  -v "$PROJECT:/workspace/project:rw" \
  -v "$PROJECT/generated_datasets/defects_rgbd_4ch_det:/dataset:rw" \
  --entrypoint python3 "$IMAGE" \
  -u \
  /workspace/project/utils/train_rgbd_4ch_yolo.py \
  --verify-loader-only \
  --data /dataset/data.yaml \
  --imgsz 640

docker rm -f "$CONTAINER" 2>/dev/null || true

echo "=== Старт RGB-D (4ch) device=$DEVICE fraction=$FRACTION workers=$WORKERS name=$NAME batch=$BATCH no_amp=${NO_AMP:-0} | docker logs -f $CONTAINER ==="
docker run -d \
  --name "$CONTAINER" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 \
  "${DOCKER_OMP_TRAIN[@]}" \
  -w /workspace/project \
  -v "$PROJECT:/workspace/project:rw" \
  -v "$PROJECT/generated_datasets/defects_rgbd_4ch_det:/dataset:rw" \
  -v "$PROJECT/yolo_runs:/runs:rw" \
  --entrypoint python3 "$IMAGE" \
  -u \
  /workspace/project/utils/train_rgbd_4ch_yolo.py \
  --data /dataset/data.yaml \
  --model /ultralytics/yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience 10 \
  --plots \
  --project /runs \
  --name "$NAME" \
  --fraction "$FRACTION" \
  "${NO_AMP_ARGS[@]}"

echo "Контейнер: $CONTAINER"
docker ps --filter "name=$CONTAINER" --format '{{.Names}} {{.Status}}'
