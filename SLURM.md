# R-Zero on Slurm

This branch follows `agent-0`'s shared cache/environment/storage layout and
configuration-driven `sbatch` wrapper. Jobs run on **one node**. The default is
two GPUs: one questioner training GPU and one solver service GPU. Solver training
uses both GPUs. Generation, grading and evaluation run sequentially on allocated
GPUs; the eight curriculum shards remain independent of the GPU count.

The upstream trainer hardcodes BF16 and FlashAttention 2, so use **A100, H100 or
H200**, preferably >=40 GB per GPU for the 4B model. Rohan's V100 GPUs are not
supported by this backend. Preflight rejects them. Capacity for a particular model
and rollout length still needs to be measured in a GPU job. Do not copy Agent0's
V100 environment: the two projects use different trainer stacks.

## Configure and install

From the R-Zero repository on a login node:

```bash
cp cluster.example.sh cluster.local.sh
# Edit shared scratch paths and Conda init/module settings in cluster.local.sh.
# The Gondor profile defaults to a100; h100 and h200 are also available.
bash scripts/submit.sh install --time=04:00:00
```

The local file is ignored by Git. All nodes must see the repository, scratch,
cache and environment paths. `RZERO_CLUSTER=gondor` selects partition, GRES and
constraint after loading the local file. For another cluster, leave
`RZERO_CLUSTER` unset and supply `RZERO_PARTITION`, `RZERO_GRES` and optionally
`RZERO_CONSTRAINT`. Optional settings include `RZERO_ACCOUNT`, `RZERO_QOS`,
`RZERO_MEM`, `RZERO_TIME`, `RZERO_CPUS` and `RZERO_ENV`.

Installation creates a Python 3.10 Conda environment in
`$REPRO_ENVS/rzero`, installs the upstream pinned dependencies plus `stopit`, then
builds FlashAttention against installed PyTorch. A CUDA toolkit/compiler compatible
with PyTorch 2.7.0 must be available on the install node; load the site's CUDA
module in `cluster.local.sh` if needed. The job needs internet access and enough
scratch space for the environment, model downloads and checkpoints. Installation
is for a fresh environment; remove or choose a different environment path before
recreating it. To use an existing compatible environment, set `RZERO_ENV` and
skip installation.

Public models need no API keys. For gated models, export `HF_TOKEN` before
submission. Logging defaults to console. No dataset/model upload is performed.
The legacy `tokens.json` fallback is accepted by the trainer, but environment
credentials take precedence and placeholder values are ignored. Keep real tokens
out of Git.

```bash
bash scripts/submit.sh preflight --time=00:10:00
```

Preflight checks backend imports and BF16 CUDA matrix multiplication on every
visible GPU. It does not validate distributed training or guarantee model fit.
Logs are written under `logs/rzero-KIND-JOBID.{out,err}`. Monitor with
`squeue -u "$USER"`; cancel with `scancel JOBID`.

## Run

Start with a small end-to-end run. Filtering may retain fewer than the required
32 questions; increase samples in that case.

```bash
bash scripts/submit.sh reproduce --time=02:00:00 -- \
  --model Qwen/Qwen3-4B-Base --name qwen3-smoke \
  --iterations 1 --questioner-steps 1 --solver-steps 1 \
  --shards 2 --samples-per-shard 100 --skip-evaluation
```

Then run the full adaptation:

```bash
bash scripts/submit.sh reproduce -- \
  --model Qwen/Qwen3-4B-Base --name qwen3-4b --iterations 5
```

`bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b` is a compatibility wrapper for
that submission. Within an existing Slurm allocation it runs directly. The
original `questioner_train*.sh`, `solver_train.sh`, `vllm_service_init/start.sh`
and `evaluation/evaluate.bash` are legacy eight-GPU scripts; the Slurm driver
does not call them.

The driver preserves `CUDA_VISIBLE_DEVICES` supplied by Slurm, including GPU UUIDs;
it never enumerates GPUs outside the allocation. It waits for solver readiness,
checks subprocess exit codes, and terminates only its own child process groups.
The upstream idle GPU-burning loop is disabled by default. Services bind to
localhost; their ports default to a job-derived range. Set `RZERO_SERVICE_PORT`
to override the first port and `RZERO_STARTUP_TIMEOUT` to override startup seconds.

Defaults are five questioner updates, fifteen solver updates, rollout batch 32,
global training batch 16, four rollout samples and tensor parallelism 1. These
reduce the GPU count and differ from the paper's batch/schedule settings.
`--service-gpus N` reserves N GPUs for solver services during questioner training;
at least one GPU must remain for training. Both training GPU counts must divide
16. Ray needs at least `5 * allocated_GPU_count + 3` CPUs, so request 43 or more
CPUs if using eight GPUs. Override submission resources before `--`, for example
`--gres=gpu:a100:8 --cpus-per-task=48`, and use `--service-gpus 4` after `--`.
All arguments before `--` are passed to `sbatch`; arguments after it go to the driver.

Source/result question shards remain in `$STORAGE_PATH/generated_question`.
Questions with majority scores in [0.3, 0.8] become local Parquet curricula in
`$STORAGE_PATH/datasets`. An empty, missing or undersized curriculum fails the job.
Checkpoints and merged models are under `$STORAGE_PATH/models`.

Completed iterations can be continued in a later job:

```bash
bash scripts/submit.sh reproduce -- --name qwen3-4b --start-iteration 2 --iterations 5
```

Use the same storage, name and step settings as the earlier run. This requires
both merged checkpoints from the preceding iteration. Incomplete iterations are
not automatically resumed: inspect and move their model directories before
restarting that iteration. Existing stage directories cause a failure to prevent
accidental overwrites. There is no automatic wall-time continuation or multi-node
support; request enough time per job or continue at a completed iteration boundary.

## Evaluation without GPT

Every completed solver iteration is evaluated unless `--skip-evaluation` is set.
To evaluate an existing model separately:

```bash
bash scripts/submit.sh evaluate -- --model /shared/path/to/actor/huggingface
```

Math benchmarks use `math_verify`; SuperGPQA/MMLU-Pro use option matching and BBEH
uses the upstream local matcher. Tensor parallelism defaults to one GPU in the
three general evaluation scripts and can be overridden through their own CLI.
Outputs go under `$STORAGE_PATH/evaluation`, and summaries append to the
repository's ignored `final_results.jsonl`. Math summaries identify their grader.

This path never invokes `results_recheck.py`. Its scores use a different procedure
from the repository's GPT-4o recheck and may be lower. The original GPT recheck
script remains available for separately configured comparisons.

## Local checks

```bash
python -m unittest discover -s tests -v
bash -n project-env.sh scripts/submit.sh scripts/main.sh slurm/job.sbatch
```

The unit tests use mocked submission/GPU/training calls. An actual GPU preflight
and the smoke run above are required to validate your cluster environment.
