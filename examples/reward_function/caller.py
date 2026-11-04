# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index, path):
    base_port = int(os.environ.get('RZERO_SERVICE_PORT', '5000'))
    response = requests.get(
        f"http://127.0.0.1:{base_port + index}/hello", params={'name': path},
        timeout=float(os.environ.get('RZERO_REQUEST_TIMEOUT', '3600')))
    response.raise_for_status()


def generate_results(data):
    count = int(os.environ.get('RZERO_SERVICE_COUNT', '4'))
    if count < 1:
        raise ValueError('RZERO_SERVICE_COUNT must be positive')
    chunks = split_list(data, count)
    paths = [generate_temp_filename(prefix=f"temp_{i}") for i in range(count)]
    try:
        for path, chunk in zip(paths, chunks):
            with open(path, 'w') as output:
                json.dump(chunk, output)
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(fetch, i, path) for i, path in enumerate(paths)]
            for future in futures:
                future.result()
        results = []
        # Preserve input ordering irrespective of HTTP completion order.
        for path in paths:
            with open(path.replace('.json', '_results.json')) as source:
                results.extend(json.load(source))
        return results
    finally:
        for path in paths:
            for candidate in (path, path.replace('.json', '_results.json')):
                if os.path.exists(candidate):
                    os.remove(candidate)


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    scores = [{"overall": min(item["score"],1-item["score"]) if item['question'] else -1,"format": 1 if item['question'] else 0,"accuracy": 1 if item['answer'] else 0} for item in final_results]
    return scores








