

Build docker image
```bash
docker build -t patch_classifier --build-context rootcontext=.. .
```

Prepare directories
```bash
touch application.log
mkdir -p tmp
```

Start collector to save traces
```bash
docker compose -f docker-compose.collect.yml up
```

Run evaluation
```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/tmp,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/../data/output,target=/workspace/data/output \
  --mount type=bind,source="$(pwd)"/application.log,target=/workspace/application.log \
  patch_classifier python -u main.py \
    --dataset fals3/cvevc_candidates \
    --subset PatchFinder_top10 \
    --split test  \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --trace \
    --collector-endpoint http://localhost:4318/v1/traces \
    --openai-api-endpoint http://localhost:8000/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp \
    --log-file application.log
```

