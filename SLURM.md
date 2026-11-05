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

## Familiar Agent0 commands

The login-node workflow uses the same entrypoint names as the sibling Agent0
checkout. Run scripts from this repository; job records and logs belong to this
checkout. Existing `cluster.local.sh` files are preserved.

```bash
# Check Rohan's requested resources without installing or allocating GPUs:
./scripts/rohan.sh install-envs --test-only

# On a supported backend, install and inspect that job:
./scripts/gondor.sh --gpu a100 install-envs
./scripts/logs.sh --install

# After installation succeeds, check the backend and download the model:
./scripts/gondor.sh preflight
./scripts/logs.sh --preflight
./scripts/gondor.sh download

# Submit a small complete iteration, then inspect its logs:
./scripts/gondor.sh training-smoke
./scripts/logs.sh --smoke

# Submit the experiment; logs.sh defaults to the latest reproduction job:
./scripts/gondor.sh reproduce -- --model Qwen/Qwen3-4B-Base --name qwen3-4b
./scripts/logs.sh
./scripts/attach.sh --gpu
```

Use the entrypoint matching the cluster you are logged into. `rohan.sh` requests
V100s with `gpu-32G`; `gondor.sh` requests A100s by default, with H100/H200
available through `--gpu`. These select scheduling settings only. **The Rohan
V100 backend port is still pending**, so the scheduling check is the ready-to-run
Rohan installation example. `preflight` retains R-Zero's backend check and needs
an installed environment, unlike Agent0's environment-independent hardware check.

| Task | Command |
| --- | --- |
| Install environment | `./scripts/install-envs.sh` or `./scripts/submit.sh install-envs` |
| View installation logs | `./scripts/logs.sh --install` |
| View a log snapshot | `./scripts/logs.sh JOB_ID --kind install-envs --once` |
| View backend check / smoke logs | `./scripts/logs.sh --preflight` / `./scripts/logs.sh --smoke` |
| View evaluation logs | `./scripts/logs.sh --eval` |
| Open a shell in a running job | `./scripts/attach.sh [JOB_ID]` |
| Monitor GPUs in an installation job | `./scripts/attach.sh --install --gpu` |
| Run a command in a running job | `./scripts/attach.sh JOB_ID -- htop` |
| Allocate an interactive shell | `./scripts/rohan.sh allocate` or `./scripts/02-allocate-gpus.sh` |
| Load Conda in that shell | `source scripts/03-load-modules.sh` |
| Download the base model | `./scripts/06-download-model.sh [--model MODEL_ID] [--revision COMMIT]` |
| Evaluate / reproduce with driver arguments | `./scripts/12-evaluate.sh` / `./scripts/14-reproduce.sh` |

`submit.sh` prints the numeric job ID to stdout and log paths to stderr, and saves
`logs/KIND-job-id.txt`. The old `install` kind aliases `install-envs`; new install
logs use `rzero-install-envs-JOBID.{out,err}`. Failed submissions and `--test-only`
leave the previous job records unchanged. `logs.sh` needs no Slurm connection or
Conda environment; it follows both logs and waits for queued jobs to start. Add
`--once` for the last 50 lines. `attach.sh` requires a RUNNING job and opens an
overlapping one-CPU step in the existing allocation. Leaving either monitor does
not cancel the batch job.

The download helper uses `$RZERO_ENV/bin/python`, writes a local download log and
records the resolved snapshot path/revision in ignored `locks/base-model.json`.
It does not require GPUs. Pass the recorded snapshot path as `--model` to keep a
specific revision fixed; a download with a custom revision does not change driver
defaults. Numbered evaluation/reproduction helpers accept driver arguments and
submit from the login node or run directly inside an allocation.

Agent0's `--resume`, `--run-dir` and `--continue-after` are not implemented by this
driver. Resume only at a completed iteration boundary with `--start-iteration`
as described below; no partial-iteration recovery is implied by the shared script
names. The original eight-GPU training scripts remain legacy entrypoints.

## Configure and install

From the R-Zero repository on a login node:

```bash
cp -n cluster.example.sh cluster.local.sh
# For a new Rohan configuration, use cluster.rohan.example.sh instead.
# Edit shared scratch paths and Conda init/module settings in cluster.local.sh.
# The Gondor profile defaults to a100; h100 and h200 are also available.
bash scripts/submit.sh install-envs --time=04:00:00
```

The local file is ignored by Git. All nodes must see the repository, scratch,
cache and environment paths. `RZERO_CLUSTER=rohan` or `gondor` selects partition,
GRES and constraint after loading the local file. Named entrypoints take
precedence over stale profile/GPU choices in that file, and preserve their
selection inside the batch allocation. For another cluster, leave
`RZERO_CLUSTER` unset and supply `RZERO_PARTITION`, `RZERO_GRES` and optionally
`RZERO_CONSTRAINT`. Optional settings include `RZERO_ACCOUNT`, `RZERO_QOS`,
`RZERO_MEM`, `RZERO_TIME`, `RZERO_CPUS`, `RZERO_MAIL_USER` and `RZERO_ENV`.

Installation and preflight request two CPUs by default, independently of
`RZERO_CPUS`, which controls reproduction/evaluation. Override with
`RZERO_INSTALL_CPUS`, `RZERO_PREFLIGHT_CPUS` or a command-line
`--cpus-per-task=N` before `--`. Installation also limits Ninja build workers to
the allocated CPUs. For `QOSMaxCpuPerUserLimit`, check your running allocations
with `squeue -u "$USER"` and the CPU limit in your effective QoS; two CPUs still
require available headroom. To check submission without starting installation:

```bash
bash scripts/rohan.sh install-envs --test-only --cpus-per-task=2 --time=04:00:00
```

This checks Slurm scheduling only; it does not validate the GPU backend. The
V100 restriction above still applies to Rohan even if the resource check passes.
Installation defaults to 32G/4 hours and preflight to 8G/10 minutes. Override
`RZERO_INSTALL_MEM`, `RZERO_INSTALL_TIME`, `RZERO_PREFLIGHT_MEM` and
`RZERO_PREFLIGHT_TIME`, or pass normal sbatch options. Interactive allocation
defaults to two CPUs, 32G and two hours, independently of the training budget;
override `RZERO_ALLOCATE_CPUS`, `RZERO_ALLOCATE_MEM`, `RZERO_ALLOCATE_TIME` or
pass srun options. Training still defaults to 16 CPUs/128G/24 hours; the current
two-GPU Ray driver requires at least 13 CPUs. Do not apply Agent0's two-CPU
training setting to it.

Installation creates a Python 3.10 Conda environment in
`$REPRO_ENVS/rzero`, installs the upstream pinned dependencies plus `stopit`, then
builds FlashAttention against installed PyTorch. A CUDA toolkit/compiler compatible
with PyTorch 2.7.0 must be available on the install node; load the site's CUDA
module in `cluster.local.sh` if needed. The job needs internet access and enough
scratch space for the environment, model downloads and checkpoints. Installation
reuses an existing Python 3.10 environment when retrying a failed package install,
rather than recreating its prefix. A different Python version requires a new
`RZERO_ENV` path. To use an existing compatible environment, set `RZERO_ENV` and
skip installation.

The cluster dependency list pins PyAV to `av==14.2.0` and the installer requires
a binary wheel for it. The upstream `14.4.0` release has only a source archive on
[PyPI](https://pypi.org/project/av/14.4.0/#files), which fails on nodes without
FFmpeg development libraries. Version `14.2.0` provides a
[Python 3.10 Linux wheel](https://pypi.org/project/av/14.2.0/#files). This is an
installation compatibility change; GPU backend validation remains required.

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
bash -n project-env.sh
for script in scripts/*.sh slurm/*.sbatch; do bash -n "$script"; done
```

The unit tests use mocked submission/GPU/training calls. An actual GPU preflight
and the smoke run above are required to validate your cluster environment.
