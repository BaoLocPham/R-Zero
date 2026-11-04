"""Build a local Parquet curriculum, retaining source shards for inspection."""
import json


def curate(storage, name, shards):
    from datasets import Dataset
    rows = []
    for shard in range(shards):
        path = storage / 'generated_question' / f'{name}_{shard}_results.json'
        for item in json.loads(path.read_text()):
            if 0.3 <= item['score'] <= 0.8 and item['answer'] not in ('', 'None', None):
                rows.append({'problem': item['question'], 'answer': item['answer'], 'score': item['score']})
    if len(rows) < 32:
        raise ValueError(f'Only {len(rows)} retained questions; need >=32 for a training rollout batch')
    path = storage / 'datasets' / f'{name}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f'Curriculum: {len(rows)} questions -> {path}', flush=True)
    return path
