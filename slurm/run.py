"""Single-node Slurm driver. Child groups are owned and cleaned up by this job."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

CHILDREN = []
PYTHON = sys.executable


def launch(args, env=None):
    print('+', ' '.join(map(str, args)), flush=True)
    child = subprocess.Popen(list(map(str, args)), env={**os.environ, **(env or {})}, start_new_session=True)
    CHILDREN.append(child)
    return child


def stop(child):
    # Also terminate workers remaining in the process group after the parent exits.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def run(args, env=None):
    child = launch(args, env)
    try:
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, args)
    finally:
        stop(child)
        CHILDREN.remove(child)


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def gpu_ids():
    import torch
    count = torch.cuda.device_count()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    ids = visible.split(',') if visible else [str(i) for i in range(count)]
    if not count or len(ids) != count:
        raise RuntimeError('GPU visibility does not match the Slurm allocation')
    return ids


def checkpoint(storage, name, step):
    return storage / 'models' / name / f'global_step_{step}' / 'actor'


def train(model, name, storage, ids, steps, questioner=False, dataset=None):
    args = [PYTHON, '-m', 'verl.trainer.main', 'config=examples/config.yaml',
            f'worker.actor.model.model_path={model}', f'trainer.experiment_name={name}',
            f'trainer.save_checkpoint_path={storage / "models" / name}',
            f'trainer.n_gpus_per_node={len(ids)}', 'trainer.nnodes=1',
            f'trainer.max_steps={steps}', f'trainer.save_freq={steps}', 'trainer.total_epochs=1000',
            'trainer.logger=[console]', 'trainer.val_before_train=false', 'trainer.val_freq=-1',
            'worker.rollout.tensor_parallel_size=1', 'worker.rollout.n=4',
            'worker.actor.global_batch_size=16', 'data.rollout_batch_size=32',
            'worker.actor.micro_batch_size_per_device_for_update=1',
            'worker.actor.micro_batch_size_per_device_for_experience=1', 'data.max_response_length=4096',
            f'data.format_prompt=./examples/format_prompt/{"questioner" if questioner else "solver"}.jinja']
    if questioner:
        args += ['worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score']
    else:
        args += [f'data.train_files={dataset}', f'data.val_files={dataset}']
    run(args, {'CUDA_VISIBLE_DEVICES': ','.join(ids)})
    actor = checkpoint(storage, name, steps)
    run([PYTHON, 'scripts/model_merger.py', '--local_dir', actor], {'CUDA_VISIBLE_DEVICES': ','.join(ids)})
    result = actor / 'huggingface'
    if not (result / 'config.json').is_file():
        raise RuntimeError(f'Merged checkpoint missing: {result}')
    return str(result)


def questioner(solver, model, name, storage, ids, steps, service_count):
    servers = []
    base_port = int(os.environ.get('RZERO_SERVICE_PORT', '5000'))
    os.environ['RZERO_SERVICE_COUNT'] = str(service_count)
    os.environ['RZERO_SERVICE_PORT'] = str(base_port)
    try:
        for i in range(service_count):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', base_port + i))
        for i, gpu in enumerate(ids[-service_count:]):
            servers.append(launch([PYTHON, 'vllm_service_init/start_vllm_server.py',
                                   '--model_path', solver, '--port', str(base_port + i)],
                                  {'CUDA_VISIBLE_DEVICES': gpu}))
        deadline = time.monotonic() + int(os.environ.get('RZERO_STARTUP_TIMEOUT', '1200'))
        for i, server in enumerate(servers):
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Solver service {i} exited: {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{base_port+i}/health', timeout=2) as response:
                        status = json.load(response)
                        if response.status == 200 and status.get('pid') == server.pid:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Solver service startup timed out')
                time.sleep(2)
        return train(model, name, storage, ids[:-service_count], steps, questioner=True)
    finally:
        for server in servers:
            stop(server)
            CHILDREN.remove(server)


def curriculum(model, solver, name, storage, ids, shards, samples):
    # Sequential shards preserve the upstream 8 x 1000 curriculum size on smaller allocations.
    for shard in range(shards):
        env = {'CUDA_VISIBLE_DEVICES': ids[shard % len(ids)]}
        run([PYTHON, '-m', 'question_generate.question_generate', '--model', model,
             '--suffix', shard, '--num_samples', samples, '--save_name', name], env)
        run([PYTHON, 'question_evaluate/evaluate.py', '--model', solver,
             '--suffix', shard, '--save_name', name], env)
    from slurm.curate import curate
    return curate(storage, name, shards)


def evaluate(model, storage, ids):
    for dataset in ('math', 'gsm8k', 'amc', 'minerva', 'olympiad', 'aime2024', 'aime2025'):
        run([PYTHON, '-m', 'evaluation.generate', '--model', model, '--dataset', dataset],
            {'CUDA_VISIBLE_DEVICES': ids[0]})
    # Persist local scores without invoking a paid judge.
    with open('final_results.jsonl', 'a') as output:
        for dataset in ('math', 'gsm8k', 'amc', 'minerva', 'olympiad', 'aime2024', 'aime2025'):
            path = storage / 'evaluation' / model.replace('/', '_') / f'results_{dataset}.json'
            results = json.loads(path.read_text())
            output.write(json.dumps({'model': model, 'dataset': dataset,
                                     'score': round(results[-1]['average_score'] * 100, 2),
                                     'grader': 'math_verify'}) + '\n')
    for dataset in ('supergpqa', 'bbeh', 'mmlupro'):
        output = storage / 'evaluation' / model.replace('/', '_') / f'{dataset}.json'
        run([PYTHON, f'evaluation/eval_{dataset}.py', '--model_path', model, '--output_file', output],
            {'CUDA_VISIBLE_DEVICES': ids[0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['reproduce', 'evaluate'])
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--name', default='qwen3-4b')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--start-iteration', type=int, default=1)
    parser.add_argument('--questioner-steps', type=int, default=5)
    parser.add_argument('--solver-steps', type=int, default=15)
    parser.add_argument('--service-gpus', type=int, default=1)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--samples-per-shard', type=int, default=1000)
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    for key in ('iterations', 'start_iteration', 'questioner_steps', 'solver_steps', 'service_gpus', 'shards', 'samples_per_shard'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if '/' in args.name or args.name in ('.', '..'):
        parser.error('--name must be a single directory name')
    storage = Path(os.environ['STORAGE_PATH']).resolve()
    ids = gpu_ids()
    if args.kind == 'evaluate':
        evaluate(args.model, storage, ids)
        return
    if len(ids) <= args.service_gpus:
        parser.error('Reproduction needs at least one training GPU plus the service GPUs')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '16'))
    if cpus < 5 * len(ids) + 3:
        parser.error(f'Ray needs at least {5 * len(ids) + 3} CPUs for {len(ids)} training GPUs')
    training_gpus = len(ids) - args.service_gpus
    if 16 % training_gpus or 16 % len(ids):
        parser.error('Both training GPU counts must divide global batch size 16')
    if args.start_iteration > args.iterations:
        parser.error('--start-iteration exceeds --iterations')
    solver = question_model = args.model
    if args.start_iteration > 1:
        previous = args.start_iteration - 1
        solver = str(checkpoint(storage, f'{args.name}_solver_v{previous}', args.solver_steps) / 'huggingface')
        question_model = str(checkpoint(storage, f'{args.name}_questioner_v{previous}', args.questioner_steps) / 'huggingface')
        for model in (solver, question_model):
            if not (Path(model) / 'config.json').is_file():
                parser.error(f'Previous iteration checkpoint missing: {model}')
    for iteration in range(args.start_iteration, args.iterations + 1):
        qname = f'{args.name}_questioner_v{iteration}'
        sname = f'{args.name}_solver_v{iteration}'
        for name in (qname, sname):
            if (storage / 'models' / name).exists():
                raise FileExistsError(f'{name} already exists; choose a new --name or move the incomplete stage before restarting')
        question_model = questioner(solver, question_model, qname, storage, ids,
                                    args.questioner_steps, args.service_gpus)
        dataset = curriculum(question_model, solver, sname, storage, ids, args.shards, args.samples_per_shard)
        solver = train(solver, sname, storage, ids, args.solver_steps, dataset=dataset)
        if not args.skip_evaluation:
            evaluate(solver, storage, ids)


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        main()
    finally:
        for child in reversed(CHILDREN):
            stop(child)
