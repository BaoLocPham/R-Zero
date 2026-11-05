"""Exercise the shell workflow with isolated storage and fake Slurm commands."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rzero scripts ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('scripts', 'clusters'):
            shutil.copytree(ROOT / name, self.root / name)
        shutil.copy(ROOT / 'project-env.sh', self.root)
        (self.root / 'slurm').mkdir()
        shutil.copy(ROOT / 'slurm/job.sbatch', self.root / 'slurm')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.logs = self.root / 'logs'
        self.logs.mkdir()
        excluded = ('RZERO_', 'REPRO_', 'SLURM_', 'BASH_FUNC_')
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(excluded)}
        self.env.update(
            PATH=str(self.bin) + ':' + os.environ['PATH'],
            REPRO_SCRATCH=str(self.root / 'scratch'),
            REPRO_CACHE=str(self.root / 'scratch/cache'),
            REPRO_ENVS=str(self.root / 'scratch/envs'),
            STORAGE_PATH=str(self.root / 'scratch/artifacts/rzero'),
            HF_HOME=str(self.root / 'scratch/cache/huggingface'),
            MOCK_CAPTURE=str(self.root / 'capture.json'),
            MOCK_SRUN_CAPTURE=str(self.root / 'srun.json'),
        )
        self.stub('sbatch', """
import json, os, sys
from pathlib import Path
Path(os.environ['MOCK_CAPTURE']).write_text(json.dumps({
    'args': sys.argv[1:],
    'profile': {key: value for key, value in os.environ.items()
                if key.startswith('RZERO_')}
}))
if '--test-only' not in sys.argv:
    print(os.environ.get('MOCK_RESULT', '417;rohan'))
sys.exit(int(os.environ.get('MOCK_EXIT', '0')))
""")

    def stub(self, name, body):
        target = self.bin / name
        target.write_text('#!/usr/bin/env python3\n' + body)
        target.chmod(0o755)
        return target

    def run_script(self, name, *args, check=True, **settings):
        result = subprocess.run([str(self.root / 'scripts' / name), *args],
                                env={**self.env, **settings}, cwd=self.root,
                                capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return result

    def capture(self):
        return json.loads((self.root / 'capture.json').read_text())

    def test_install_alias_saves_canonical_and_legacy_job_records(self):
        result = self.run_script('submit.sh', 'install')
        self.assertEqual(result.stdout, '417\n')
        args = self.capture()['args']
        self.assertIn('--job-name=rzero-install-envs', args)
        self.assertIn('--cpus-per-task=2', args)
        self.assertIn('--mem=32G', args)
        self.assertIn('--time=04:00:00', args)
        self.assertEqual(args[-1], 'install-envs')
        for kind in ('install-envs', 'install'):
            self.assertEqual((self.logs / f'{kind}-job-id.txt').read_text(), '417\n')
        self.assertIn('rzero-install-envs-417.{out,err}', result.stderr)

    def test_failed_submission_preserves_previous_job_and_status(self):
        record = self.logs / 'reproduce-job-id.txt'
        record.write_text('123\n')
        result = self.run_script('submit.sh', 'reproduce', check=False,
                                 MOCK_EXIT='7', MOCK_RESULT='')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(record.read_text(), '123\n')
        self.assertIn('No job ID was saved', result.stderr)

    def test_test_only_preserves_records_and_does_not_claim_a_job(self):
        record = self.logs / 'install-envs-job-id.txt'
        record.write_text('123\n')
        result = self.run_script('rohan.sh', 'install-envs', '--test-only')
        self.assertEqual(result.stdout, '')
        self.assertIn('No job was submitted', result.stderr)
        self.assertEqual(record.read_text(), '123\n')
        self.assertFalse((self.logs / 'install-job-id.txt').exists())

    def test_invalid_submission_output_is_not_saved(self):
        result = self.run_script('submit.sh', 'reproduce', check=False,
                                 MOCK_RESULT='Submitted batch job 417')
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.logs / 'reproduce-job-id.txt').exists())
        self.assertIn('inspect the queue', result.stderr)

    def test_named_cluster_overrides_stale_local_profile_and_survives_resourcing(self):
        (self.root / 'cluster.local.sh').write_text(
            'export RZERO_CLUSTER=gondor\n'
            'export RZERO_GONDOR_GPU=a100\n'
            'export RZERO_GRES=gpu:a100:8\n'
            'export RZERO_CONSTRAINT=gpu-a100\n')
        self.run_script('rohan.sh', 'preflight')
        capture = self.capture()
        self.assertIn('--gres=gpu:v100:2', capture['args'])
        self.assertIn('--constraint=gpu-32G', capture['args'])
        result = subprocess.run(
            ['bash', '-c', 'source ./project-env.sh; printf "%s\\n" "$RZERO_CLUSTER" "$RZERO_GRES"'],
            cwd=self.root, env={**self.env, **capture['profile']},
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'rohan\ngpu:v100:2\n')
        (self.root / 'cluster.local.sh').write_text(
            'export RZERO_CLUSTER=rohan\n'
            'export RZERO_GONDOR_GPU=a100\n'
            'export RZERO_GRES=gpu:v100:2\n'
            'export RZERO_CONSTRAINT=gpu-32G\n')
        self.run_script('gondor.sh', '--gpu', 'h100', 'reproduce')
        capture = self.capture()
        self.assertIn('--gres=gpu:h100:2', capture['args'])
        self.assertIn('--partition=gpu-large', capture['args'])
        self.assertIn('--constraint=gpu-h100', capture['args'])
        result = subprocess.run(
            ['bash', '-c', 'source ./project-env.sh; printf "%s\\n" "$RZERO_CLUSTER" "$RZERO_GRES"'],
            cwd=self.root, env={**self.env, **capture['profile']},
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'gondor\ngpu:h100:2\n')

    def test_cluster_wrapper_preserves_resource_and_model_arguments(self):
        self.run_script('gondor.sh', '--gpu=h200', 'reproduce', '--time=01:00:00', '--',
                        '--model', '/shared/model with spaces', '--iterations', '1')
        args = self.capture()['args']
        self.assertIn('--gres=gpu:h200:2', args)
        self.assertIn('--time=01:00:00', args)
        self.assertEqual(args[-5:], ['reproduce', '--model', '/shared/model with spaces',
                                    '--iterations', '1'])

    def test_invalid_gpu_or_agent0_continuation_never_submits(self):
        for args in (('gondor.sh', '--gpu', 'v100', 'install-envs'),
                     ('rohan.sh', 'reproduce', '--continue-after', '417'),
                     ('rohan.sh', 'reproduce', '--', '--resume'),
                     ('rohan.sh', 'reproduce', '--', '--run-dir=/shared/run')):
            with self.subTest(args=args):
                result = self.run_script(*args, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertFalse((self.root / 'capture.json').exists())

    def test_numbered_helpers_forward_driver_arguments(self):
        for script, kind in (('12-evaluate.sh', 'evaluate'), ('14-reproduce.sh', 'reproduce')):
            with self.subTest(script=script):
                self.run_script(script, '--model', '/shared/model with spaces')
                self.assertEqual(self.capture()['args'][-3:],
                                 [kind, '--model', '/shared/model with spaces'])

    def test_log_snapshot_reads_both_files_without_loading_cluster_settings(self):
        (self.root / 'cluster.local.sh').write_text('echo "Must not load config" >&2; return 2\n')
        (self.logs / 'reproduce-job-id.txt').write_text('417\n')
        (self.logs / 'rzero-reproduce-417.out').write_text('training stdout\n')
        (self.logs / 'rzero-reproduce-417.err').write_text('training stderr\n')
        result = self.run_script('logs.sh', '--once')
        self.assertIn('training stdout', result.stdout)
        self.assertIn('training stderr', result.stdout)
        self.assertNotIn('Must not load config', result.stderr)
        self.assertFalse((self.root / 'capture.json').exists())

    def test_install_log_selection_and_old_log_names(self):
        (self.logs / 'install-envs-job-id.txt').write_text('417\n')
        (self.logs / 'rzero-install-envs-417.out').write_text('new install stdout\n')
        (self.logs / 'rzero-install-envs-417.err').write_text('new install stderr\n')
        result = self.run_script('logs.sh', '--install', '--once')
        self.assertIn('new install stdout', result.stdout)
        self.assertIn('new install stderr', result.stdout)
        (self.logs / 'install-envs-job-id.txt').unlink()
        (self.logs / 'install-job-id.txt').write_text('123\n')
        (self.logs / 'rzero-install-123.out').write_text('old install stdout\n')
        (self.logs / 'rzero-install-123.err').write_text('old install stderr\n')
        result = self.run_script('logs.sh', '--install', '--once')
        self.assertIn('old install stdout', result.stdout)
        self.assertIn('old install stderr', result.stdout)

    def test_log_follow_waits_for_queued_job_files(self):
        self.stub('tail', 'import json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        (self.logs / 'training-smoke-job-id.txt').write_text('417\n')
        result = self.run_script('logs.sh', '--smoke')
        self.assertIn('Waiting for log files', result.stdout)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[:4], ['-n', '50', '-F', '--'])
        self.assertIn(str(self.logs / 'rzero-training-smoke-417.err'), args)

    def test_log_missing_or_invalid_record_is_actionable(self):
        result = self.run_script('logs.sh', '--once', check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn('No saved reproduce job ID', result.stderr)
        (self.logs / 'reproduce-job-id.txt').write_text('../bad\n')
        result = self.run_script('logs.sh', '--once', check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid', result.stderr)

    def setup_attach(self, state='RUNNING'):
        self.stub('squeue', 'import os,sys\n'
                            'print(os.environ.get("MOCK_STATE", "RUNNING"))\n'
                            'sys.exit(int(os.environ.get("MOCK_QUEUE_EXIT", "0")))\n')
        self.stub('srun', 'import json,os,sys\nfrom pathlib import Path\n'
                          'Path(os.environ["MOCK_SRUN_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
        (self.logs / 'reproduce-job-id.txt').write_text('417\n')
        self.env['MOCK_STATE'] = state

    def test_attach_uses_existing_allocation_and_preserves_command_arguments(self):
        self.setup_attach()
        self.run_script('attach.sh', '--', 'printf', '%s', 'argument with spaces')
        args = json.loads((self.root / 'srun.json').read_text())
        self.assertEqual(args, ['--jobid=417', '--overlap', '-N1', '-n1', '-c1', '--pty',
                                'printf', '%s', 'argument with spaces'])
        self.run_script('attach.sh', '--gpu')
        args = json.loads((self.root / 'srun.json').read_text())
        self.assertEqual(args[-4:], ['watch', '-n', '1', 'nvidia-smi'])

    def test_attach_refuses_pending_jobs_failed_queries_and_conflicting_commands(self):
        self.setup_attach('PENDING')
        result = self.run_script('attach.sh', check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn('requires RUNNING', result.stderr)
        self.assertFalse((self.root / 'srun.json').exists())
        result = self.run_script('attach.sh', check=False, MOCK_QUEUE_EXIT='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('Cannot query', result.stderr)
        result = self.run_script('attach.sh', '--gpu', '--', 'htop', check=False)
        self.assertEqual(result.returncode, 2)

    def test_allocate_uses_rohan_profile_and_small_independent_cpu_budget(self):
        self.setup_attach()
        self.run_script('rohan.sh', 'allocate', '--time=00:30:00')
        args = json.loads((self.root / 'srun.json').read_text())
        self.assertIn('--gres=gpu:v100:2', args)
        self.assertIn('--constraint=gpu-32G', args)
        self.assertIn('--cpus-per-task=2', args)
        self.assertIn('--time=00:30:00', args)
        self.assertEqual(args[-3:], ['--pty', 'bash', '-l'])

    def test_module_helper_initializes_conda_shell_function(self):
        base = self.root / 'fake conda'
        initialization = base / 'etc/profile.d/conda.sh'
        initialization.parent.mkdir(parents=True)
        initialization.write_text(
            'conda() {\n'
            ' if [[ "$*" == "info --base" ]]; then printf "%s\\n" '
            + shlex.quote(str(base)) + '\n'
            ' else printf "fake conda initialized\\n"; fi\n'
            '}\n')
        result = subprocess.run(
            ['bash', '-c', 'source scripts/03-load-modules.sh; test "$(type -t conda)" = function'],
            cwd=self.root, env={**self.env, 'RZERO_CONDA_INIT': str(initialization)},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fake conda initialized', result.stdout)

    def test_module_helper_initializes_conda_after_module_only_adds_it_to_path(self):
        base = self.root / 'module conda'
        (base / 'bin').mkdir(parents=True)
        initialization = base / 'etc/profile.d/conda.sh'
        initialization.parent.mkdir(parents=True)
        initialization.write_text('conda() { printf "initialized shell function\\n"; }\n')
        executable = base / 'bin/conda'
        executable.write_text('#!/bin/bash\nprintf "%s\\n" ' + shlex.quote(str(base)) + '\n')
        executable.chmod(0o755)
        shell = (
            'module() { if [[ "$1" == load ]]; then export PATH='
            + shlex.quote(str(base / 'bin')) + ':$PATH; fi; }\n'
            'source scripts/03-load-modules.sh\n'
            'test "$(type -t conda)" = function\n')
        result = subprocess.run(
            ['bash', '-c', shell], cwd=self.root,
            env={**self.env, 'PATH': '/usr/bin:/bin', 'RZERO_CONDA_MODULE': 'FakeConda'},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('initialized shell function', result.stdout)

    def test_training_smoke_template_passes_small_run_and_explicit_overrides(self):
        prefix = self.root / 'venv'
        (prefix / 'bin').mkdir(parents=True)
        activation = prefix / 'bin/activate'
        activation.write_text('export PATH=' + shlex.quote(str(prefix / 'bin')) + ':$PATH\n')
        interpreter = prefix / 'bin/python'
        interpreter.write_text(
            '#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n'
            'with Path(os.environ["MOCK_PYTHON"]).open("a") as output:\n'
            ' output.write(json.dumps(sys.argv[1:]) + "\\n")\n')
        interpreter.chmod(0o755)
        environment = {**self.env, 'SLURM_JOB_ID': '417', 'SLURM_CPUS_PER_TASK': '16',
                       'RZERO_ROOT': str(self.root), 'RZERO_ENV': str(prefix),
                       'RZERO_CONDA_INIT': '', 'RZERO_CONDA_MODULE': '',
                       'MOCK_PYTHON': str(self.root / 'python.jsonl')}
        result = subprocess.run(
            ['bash', 'slurm/job.sbatch', 'training-smoke', '--model', '/shared/model with spaces',
             '--name', 'my-smoke'], env=environment, cwd=self.root,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root / 'python.jsonl').read_text().splitlines()]
        self.assertEqual(calls[0], ['-m', 'slurm.preflight'])
        self.assertEqual(calls[1][:3], ['-m', 'slurm.run', 'reproduce'])
        self.assertIn('--skip-evaluation', calls[1])
        self.assertEqual(calls[1][-4:], ['--model', '/shared/model with spaces', '--name', 'my-smoke'])

    def installer_environment(self):
        shutil.copy(ROOT / 'requirements.txt', self.root)
        prefix = self.root / 'partial environment'
        (prefix / 'bin').mkdir(parents=True)
        interpreter = prefix / 'bin/python'
        interpreter.write_text(
            '#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n'
            'args = sys.argv[1:]\n'
            'with Path(os.environ["MOCK_PYTHON"]).open("a") as output:\n'
            ' output.write(json.dumps({"args": args, "max_jobs": os.environ.get("MAX_JOBS"), '
            '"visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")\n'
            'if args and args[0] == "-c" and os.environ.get("MOCK_BAD_PYTHON"):\n'
            ' sys.exit(1)\n'
            'if "--only-binary=av" in args and os.environ.get("MOCK_AV_FAILURE"):\n'
            ' sys.exit(23)\n')
        interpreter.chmod(0o755)
        base = self.root / 'fake conda'
        initialization = base / 'etc/profile.d/conda.sh'
        initialization.parent.mkdir(parents=True)
        initialization.write_text(
            'conda() {\n case "$1" in\n'
            ' info) printf "%s\\n" ' + shlex.quote(str(base)) + ' ;;\n'
            ' --version) echo "mock conda" ;;\n'
            ' activate) export PATH="$2/bin:$PATH" ;;\n'
            ' create) echo "unexpected recreation" > "$MOCK_CONDA_CREATE"; return 1 ;;\n'
            ' esac\n}\n')
        return {**self.env, 'SLURM_JOB_ID': '417', 'SLURM_CPUS_PER_TASK': '2',
                'CUDA_VISIBLE_DEVICES': '0,1', 'RZERO_ROOT': str(self.root),
                'RZERO_ENV': str(prefix), 'RZERO_CONDA_INIT': str(initialization),
                'MOCK_CONDA_CREATE': str(self.root / 'conda-create.txt'),
                'MOCK_PYTHON': str(self.root / 'python.jsonl')}

    def test_install_retry_reuses_prefix_and_requires_av_wheel_in_gpu_job(self):
        environment = self.installer_environment()
        result = subprocess.run(['bash', 'slurm/job.sbatch', 'install-envs'],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Reusing existing Python 3.10 environment', result.stdout)
        self.assertFalse((self.root / 'conda-create.txt').exists())
        calls = [json.loads(line) for line in (self.root / 'python.jsonl').read_text().splitlines()]
        dependency_installs = [call for call in calls if '--only-binary=av' in call['args']]
        self.assertEqual(len(dependency_installs), 1)
        for call in calls:
            self.assertEqual(call['visible'], '0,1')
            self.assertEqual(call['max_jobs'], '2')
        generated = (Path(environment['RZERO_ENV']) / 'requirements-no-flash.txt').read_text()
        self.assertIn('av==14.2.0\n', generated)
        self.assertNotIn('av==14.4.0\n', generated)
        self.assertNotIn('flash_attn==', generated)
        self.assertEqual(calls[-1]['args'], ['-m', 'slurm.preflight'])

    def test_failed_av_install_stops_before_flash_attention_and_preflight(self):
        environment = {**self.installer_environment(), 'MOCK_AV_FAILURE': '1'}
        result = subprocess.run(['bash', 'slurm/job.sbatch', 'install-envs'],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 23)
        calls = [json.loads(line)['args']
                 for line in (self.root / 'python.jsonl').read_text().splitlines()]
        self.assertTrue(all('flash_attn==2.7.4.post1' not in call for call in calls))
        self.assertNotIn(['-m', 'slurm.preflight'], calls)

    def test_install_rejects_wrong_python_and_requires_slurm_allocation(self):
        environment = {**self.installer_environment(), 'MOCK_BAD_PYTHON': '1'}
        result = subprocess.run(['bash', 'slurm/job.sbatch', 'install-envs'],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'conda-create.txt').exists())
        calls = (self.root / 'python.jsonl').read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0])['args'][0], '-c')
        environment.pop('SLURM_JOB_ID')
        result = subprocess.run(['bash', 'slurm/job.sbatch', 'install-envs'],
                                env=environment, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Slurm allocation', result.stderr)
        self.assertEqual((self.root / 'python.jsonl').read_text().splitlines(), calls)

    def test_download_records_resolved_revision_without_network(self):
        snapshot = self.root / 'cache/snapshots/abcdef'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return str(snapshot)

        spec = importlib.util.spec_from_file_location('download_model', ROOT / 'scripts/download-model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = self.root / 'locks/base-model.json'
        argv = ['download-model.py', '--model', 'org/model', '--revision', 'abcdef',
                '--record', str(record)]
        with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(snapshot_download=download)}), \
                patch.object(sys, 'argv', argv):
            module.main()
        self.assertEqual(calls, [{'repo_id': 'org/model', 'revision': 'abcdef'}])
        saved = json.loads(record.read_text())
        self.assertEqual(saved['revision'], 'abcdef')
        self.assertEqual(saved['snapshot_path'], str(snapshot))
        self.assertEqual(list(record.parent.iterdir()), [record])


if __name__ == '__main__':
    unittest.main()
