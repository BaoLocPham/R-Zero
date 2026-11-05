"""Download once into HF_HOME and record the exact resolved snapshot."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=args.model, revision=args.revision)).resolve()
    if not (snapshot / 'config.json').is_file():
        raise RuntimeError(f'Download has no model config: {snapshot}')
    record = {'repo_id': args.model, 'requested_revision': args.revision,
              'revision': snapshot.name, 'snapshot_path': str(snapshot)}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=args.record.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(record, output, indent=2)
        output.write('\n')
    try:
        os.replace(temporary, args.record)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'PASS: downloaded {args.model} at revision {snapshot.name}')
    print(f'Snapshot: {snapshot}')
    print(f'Record: {args.record}')
    print('Use the snapshot path as --model to keep this revision fixed.')


if __name__ == '__main__':
    main()
