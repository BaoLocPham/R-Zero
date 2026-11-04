import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from slurm import run

ROOT = Path(__file__).resolve().parents[1]


class SlurmTests(unittest.TestCase):
    def test_submission_preserves_options_and_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            sbatch = Path(directory) / 'sbatch'
            sbatch.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            sbatch.chmod(0o755)
            env = {**os.environ, 'PATH': directory + ':' + os.environ['PATH'],
                   'RZERO_CLUSTER': 'gondor', 'RZERO_GONDOR_GPU': 'h100'}
            result = subprocess.run(['bash', 'scripts/submit.sh', 'reproduce', '--time=00:30:00', '--',
                                     '--model', '/shared/model with spaces', '--iterations', '1'],
                                    cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            args = json.loads(result.stdout)
            self.assertIn('--gres=gpu:h100:2', args)
            self.assertIn('--partition=gpu-large', args)
            self.assertIn('--time=00:30:00', args)
            self.assertEqual(args[-5:], ['reproduce', '--model', '/shared/model with spaces', '--iterations', '1'])

    def test_training_step_limit_across_epoch_boundaries(self):
        # Execute the production loop guards around a tiny stand-in update body.
        tree = ast.parse((ROOT / 'verl/trainer/ray_trainer.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayPPOTrainer')
        fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'fit')
        outer = next(node for node in fit.body if isinstance(node, ast.For))
        inner = next(node for node in outer.body if isinstance(node, ast.For))
        for batches in (1, 3, 20):
            state = types.SimpleNamespace(global_step=0, training_steps=5)
            loop = ast.For(target=ast.Name(id='epoch', ctx=ast.Store()),
                           iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(1000)], keywords=[]),
                           body=[ast.For(target=ast.Name(id='batch', ctx=ast.Store()),
                                         iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(batches)], keywords=[]),
                                         body=inner.body[:2], orelse=[]), outer.body[-1]], orelse=[])
            code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<loop>', 'exec')
            scope = {'self': state}
            exec(code, scope)
            self.assertEqual(state.global_step, 5)
            self.assertLess(scope['epoch'], 5)

    def test_reward_shards_preserve_input_order(self):
        from concurrent.futures import ThreadPoolExecutor
        for filename in ('caller.py', 'caller_penalty.py'):
            tree = ast.parse((ROOT / 'examples/reward_function' / filename).read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in ('split_list', 'generate_results')]
            with tempfile.TemporaryDirectory() as directory:
                counter = iter(range(10))
                def filename_for(**kwargs):
                    return str(Path(directory) / f'{next(counter)}.json')
                def fetch(index, path):
                    items = json.loads(Path(path).read_text())
                    Path(path.replace('.json', '_results.json')).write_text(json.dumps(items))
                scope = {'os': os, 'json': json, 'ThreadPoolExecutor': ThreadPoolExecutor,
                         'fetch': fetch, 'generate_temp_filename': filename_for}
                exec(compile(ast.Module(body=functions, type_ignores=[]), '<reward>', 'exec'), scope)
                with patch.dict(os.environ, {'RZERO_SERVICE_COUNT': '2'}):
                    self.assertEqual(scope['generate_results']([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cuda_mask_preserved(self):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 2))
        with patch.dict(sys.modules, {'torch': fake}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-ab,GPU-cd'}):
            self.assertEqual(run.gpu_ids(), ['GPU-ab', 'GPU-cd'])

    def test_training_gpu_isolation_and_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            merged = run.checkpoint(storage, 'test', 5) / 'huggingface'
            merged.mkdir(parents=True)
            (merged / 'config.json').write_text('{}')
            with patch.object(run, 'run') as execute:
                self.assertEqual(run.train('base', 'test', storage, ['GPU-one'], 5, True), str(merged))
            args, env = execute.call_args_list[0].args
            self.assertIn('trainer.n_gpus_per_node=1', args)
            self.assertIn('trainer.max_steps=5', args)
            self.assertIn('worker.rollout.tensor_parallel_size=1', args)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-one')

    def test_failed_child_propagates(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run.run([sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(run.CHILDREN, [])

    def test_curriculum_rejects_small_and_missing_shards(self):
        from slurm.curate import curate
        fake = types.SimpleNamespace(Dataset=object)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'datasets': fake}):
            storage = Path(directory)
            path = storage / 'generated_question'
            path.mkdir()
            with self.assertRaises(FileNotFoundError):
                curate(storage, 'test', 1)
            (path / 'test_0_results.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Only 0'):
                curate(storage, 'test', 1)

    def test_evaluation_no_judge_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            path = storage / 'evaluation' / 'org_model'
            path.mkdir(parents=True)
            for dataset in ('math', 'gsm8k', 'amc', 'minerva', 'olympiad', 'aime2024', 'aime2025'):
                (path / f'results_{dataset}.json').write_text('[{"average_score": 0.5}]')
            previous = Path.cwd()
            try:
                os.chdir(storage)
                with patch.object(run, 'run') as execute:
                    run.evaluate('org/model', storage, ['GPU-one'])
                self.assertEqual(execute.call_count, 10)
                self.assertFalse(any('results_recheck.py' in str(call) for call in execute.call_args_list))
                rows = [json.loads(line) for line in Path('final_results.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 7)
                self.assertTrue(all(row['score'] == 50 and row['grader'] == 'math_verify' for row in rows))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
