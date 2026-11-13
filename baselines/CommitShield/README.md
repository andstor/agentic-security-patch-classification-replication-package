

Build docker image
```bash
docker build -t commitshield .
```

Prepare directories
```bash
touch application.log
mkdir -p tmp
```


Run CommitShield patch classification
```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/tmp,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/../../data/baselines/commitshield/output,target=/workspace/data/output \
  commitshield python -u vul_fix_check.py \
    --subset PatchFinder_top10 \
    --token ghp_111111111111111111111111111111111111 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --openai-api-endpoint http://localhost:8000/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp
```





Run CommitShield patch classification
```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/tmp,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/../../data/baselines/commitshield/output,target=/workspace/data/output \
  commitshield python -u vul_fix_check.py \
    --subset PatchFinder_top10 \
    --token ghp_111111111111111111111111111111111111 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --openai-api-endpoint https://vllm1.andstor.dev/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp
```





```bash
python vul_fix_check.py \
  --token ghp_111111111111111111111111111111111111 \
  --openai-api-endpoint https://vllm1.andstor.dev/v1 \
  --openai-api-key sk-111111111111111111111111111111111111111111111111 \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --subset PatchFinder_top10
```



### Apptainer
If you want to use Apptainer instead of Docker, we provide pre-built images on GitHub Container Registry. The image is available at `ghcr.io/andstor/agentic-security-patch-classification-replication-package/commitshield:main`.

Because we are executing untrusted code, we recommend using the `--containall` and `--no-home` flags to prevent the container from accessing your home directory and other sensitive files. This will require the use of an overlay file to store intermediate dependencies and results.

You can create an overlay file using the following command:

```bash
apptainer overlay create --sparse --size 10240 overlay2.img
```

For more information on how to use Apptainer, please refer to the [Apptainer documentation](https://apptainer.org/docs/user/latest/).


```bash
apptainer run \
  --containall \
  --no-home \
  --overlay overlay.img \
  --cwd "/workspace/" \
  --mount type=bind,source="$(pwd)"/tmp/,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/../../data/baselines/CommitShield/output/,target=/workspace/data/output \
  docker://ghcr.io/andstor/agentic-security-patch-classification-replication-package/commitshield:main python -u vul_fix_check.py \
    --subset PatchFinder_top10 \
    --token ghp_111111111111111111111111111111111111 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --openai-api-endpoint https://vllm1.andstor.dev/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp
```
```bash
apptainer run \
  --containall \
  --no-home \
  --overlay overlay2.img \
  --cwd "/workspace/" \
  --mount type=bind,source="$(pwd)"/../../data/baselines/CommitShield/output2/,target=/workspace/data/output \
  docker://ghcr.io/andstor/agentic-security-patch-classification-replication-package/commitshield:main python -u vul_fix_check.py \
    --subset random_10 \
    --token ghp_111111111111111111111111111111111111 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --openai-api-endpoint https://vllm1.andstor.dev/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp
```









Run CommitShield patch classification
```bash
docker run \
  -it \
  --mount type=bind,source="$(pwd)"/tmp,target=/workspace/tmp \
  --mount type=bind,source="$(pwd)"/../../data/baselines/commitshield/output,target=/workspace/data/output \
  commitshield python -u vul_fix_check.py \
    --subset PatchFinder_top10 \
    --token ghp_111111111111111111111111111111111111 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --openai-api-endpoint https://vllm1.andstor.dev/v1 \
    --openai-api-key sk-111111111111111111111111111111111111111111111111 \
    --output-dir data/output \
    --local-dir tmp
```



git submodule update --init --recursive



apptainer overlay create --sparse --size 150240 overlay3.img



tune2fs -l overlay.img | grep -E 'Block count|Free blocks|Filesystem size'

4096  * 2621440 / 1e+9

lfs quota -uh andstorh /cluster/