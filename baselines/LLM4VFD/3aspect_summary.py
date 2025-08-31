from datasets import load_dataset
from utils import process_patch
from openai import OpenAI
from prompts import SYSTEM_PROMPT_CAVFD, SYSTEM_PROMPT_CCI, SYSTEM_PROMPT_DA, USER_PROMPT_CAVFD, USER_PROMPT_CCI, USER_PROMPT_DA
import os
from tqdm import tqdm
import json
from multiprocessing.dummy import Pool as ThreadPool

def process_example(args_tuple):
    """Worker function to process a single example"""
    cve, commit_id, diff, args = args_tuple

    client = OpenAI(
            api_key=args.openai_api_key,
            base_url=args.openai_api_endpoint,
        )
    try:
        user_prompt = USER_PROMPT_CCI.substitute(patch_content=diff)
        system_prompt = SYSTEM_PROMPT_CCI
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        summary = response.choices[0].message.content
        return {
            "cve": cve,
            "commit_id": commit_id,
            "three_aspect_summary": summary
        }
    except Exception as e:
        return None

def main(args):
    cve_ds = load_dataset("fals3/cvevc_cve")
    patches_ds = load_dataset("fals3/cvevc_commits", "patches")
    mappings_ds = load_dataset("fals3/cvevc_cve_commit_mappings")
    mappings_ds = mappings_ds.filter(lambda x: x["label"] == 1, num_proc=10) # keep only positive samples


    for split in ["train", "validation", "test"]:
        model_name = args.model.split("/")[1]
        output_file = os.path.join(args.output_dir, f"{model_name}_{split}.jsonl")
        if os.path.exists(output_file):
            print(f"Output file {output_file} exists, resuming generation.")
            existing_commit_ids = set()
            with open(output_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    existing_commit_ids.add(data["commit_id"])
            resumable_ds = mappings_ds[split].filter(lambda x: x["commit_id"] not in existing_commit_ids, num_proc=10)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            resumable_ds = mappings_ds[split]

        # Prepare a lookup for commit_id -> commit for this split
        pindex = {key: idx for idx, key in tqdm(enumerate(patches_ds[split]["commit_id"]), total=len(patches_ds[split]["commit_id"]), desc=f"Indexing patch commits {split}")}

        # Prepare arguments for thread pool as a generator
        def task_args_gen():
            for example in resumable_ds:
                try:
                    commit = patches_ds[split][pindex[example["commit_id"]]]
                    diff = process_patch(commit["diff"])
                    yield (example["cve"], example["commit_id"], diff, args)
                except KeyError:
                    continue

        with ThreadPool(args.num_threads) as pool:
            results = pool.imap_unordered(process_example, task_args_gen())
            with open(output_file, "a") as f:
                for result in tqdm(results, total=len(resumable_ds), desc=f"Processing {split}"):
                    if result is None:
                        continue
                    else:
                        f.write(json.dumps({
                            "cve": result["cve"],
                            "commit_id": result["commit_id"],
                            "three_aspect_summary": result["three_aspect_summary"]
                        }) + "\n")
                        f.flush()
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Compute lexical similarity for CVE-commit dataset.")
    parser.add_argument("--output_dir", type=str, default='../../data/baselines/LLM4VFD/3aspect_summaries', help="Output directory for lexical similarity CSVs")
    parser.add_argument("--openai-api-endpoint", type=str, default="http://localhost:8000/v1", help="OpenAI API compatible endpoint")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API key")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Model to use for inference")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for concurrent processing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)