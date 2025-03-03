import json
from argparse import ArgumentParser

from binoculars import Binoculars
from tqdm import tqdm
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="detection/2018-2020-comments.jsonl")
    parser.add_argument("--device-1", type=str, default="cuda:0")
    parser.add_argument("--device-2", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-token-observed", type=int, default=2048)
    parser.add_argument("--output", type=str, default="detection/2018-2020-comments-detected.jsonl")
    args = parser.parse_args()

    bino = Binoculars(device_1=args.device_1, device_2=args.device_2, max_token_observed=args.max_token_observed)

    output_path = args.output
    while os.path.exists(output_path):
        output_path = output_path.replace(".jsonl", "_1.jsonl")

    # with open(args.input, "r") as f:
    #     total_lines = sum(1 for _ in f)

    # print(f"Processing {total_lines} lines")
    total_lines = 1000000
    
    with open(args.input, "r") as f:
        data = []
        for idx, line in tqdm(enumerate(f), total=total_lines):
            entry = json.loads(line)
            if len(entry["body"].split()) <= 150:
                continue
            data.append((idx, entry["body"]))
            if len(data) == args.batch_size:
                scores = bino.compute_score([comment for _, comment in data])
                with open(output_path, "a") as f:
                    for score, (idx, comment) in zip(scores, data):
                        f.write(json.dumps({"idx": idx, "score": score, "body": comment}) + "\n")
                del scores
                data = []

    if data:
        scores = bino.compute_score([comment for _, comment in data])
        with open(output_path, "a") as f:
            for score, (idx, comment) in zip(scores, data):
                f.write(json.dumps({"idx": idx, "score": score, "body": comment}) + "\n")