from datasets import load_dataset
from utils import process_patch
from openai import OpenAI
from prompts import SYSTEM_PROMPT_CAVFD, SYSTEM_PROMPT_CCI, SYSTEM_PROMPT_DA, USER_PROMPT_CAVFD, USER_PROMPT_CCI, USER_PROMPT_DA
import os
from tqdm import tqdm
import json
from multiprocessing.dummy import Pool as ThreadPool
import chromadb
from datasets import concatenate_datasets





def process_cci(args_tuple):
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
            "diff": diff,
            "three_aspect_summary": summary
        }
    except Exception as e:
        return None
    
    
def process_embedding(args_tuple):
    """Worker function to process a single example"""

    cve, commit_id, diff, three_aspect_summary, args = args_tuple

    client = OpenAI(
            api_key=args.openai_embedding_api_key,
            base_url=args.openai_embedding_api_endpoint,
        )
    
    try:
        response = client.embeddings.create(
            input=three_aspect_summary,
            model=args.embedding_model,
        )
        embedding = response.data[0].embedding
        
        return {
            "cve": cve,
            "commit_id": commit_id,
            "diff": diff,
            "three_aspect_summary": three_aspect_summary,
            "three_aspect_summary_embedding": embedding
        }
    except Exception as e:
        return None

def process_cavfd(args_tuple):
    """Worker function to process a single example"""
    
    cve, commit_id, diff, cci, history_cci, history_cve_description, args = args_tuple
    
    client = OpenAI(
            api_key=args.openai_api_key,
            base_url=args.openai_api_endpoint,
        )
    try:
        user_prompt = USER_PROMPT_CAVFD.substitute(patch_content = diff, three_aspect_content = cci, history_three_aspect_content = history_cci, history_vuln_content = history_cve_description)
        system_prompt = SYSTEM_PROMPT_CAVFD
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        cavfd = response.choices[0].message.content
        return {
            "cve": cve,
            "commit_id": commit_id,
            "cavfd": cavfd,
        }
    
    except Exception:
        return None
    
def main(args):

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    print(chroma_client.list_collections())
    model_name = args.model.split("/")[1]
    collection = chroma_client.get_collection(name=f"3aspect_summary_{model_name}")
    
    cve_ds = load_dataset("fals3/cvevc_cve")
    cve_ds = concatenate_datasets(cve_ds.values())
    patches_ds = load_dataset("fals3/cvevc_commits", "patches")
    patches_ds = concatenate_datasets(patches_ds.values())
    cands_ds = load_dataset("fals3/cvevc_candidates", args.subset, split="test")
    
    # Prepare a lookup for cve descriptions
    cve_index = {key: idx for idx, key in tqdm(enumerate(cve_ds["cve"]), total=len(cve_ds["cve"]), desc=f"Indexing CVEs")}
    cand_index = {key: idx for idx, key in tqdm(enumerate(cands_ds["commit_id"]), total=len(cands_ds["commit_id"]), desc=f"Indexing candidate commits")}
    
    
    output_file = os.path.join(args.output_dir, f"{args.subset}_{model_name}.jsonl")
    if os.path.exists(output_file):
        print(f"Output file {output_file} exists, resuming generation.")
        existing_commit_ids = set()
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                existing_commit_ids.add(data["input"]["commit_id"])
        resumable_ds = cands_ds.filter(lambda x: x["commit_id"] not in existing_commit_ids, num_proc=10)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        resumable_ds = cands_ds



    def task_args_gen():
        for example in resumable_ds:
            try:
                diff = process_patch(example["diff"])
                yield (example["cve"], example["commit_id"], diff, args)
            except KeyError:
                continue

    def task_cci_args_gen(cci_data):
        for data in cci_data:
            if data is None:
                continue
            yield (data["cve"], data["commit_id"], data["diff"], data["three_aspect_summary"], args)

    def task_rag_args_gen(embedding_data):
        for data in embedding_data:
            if data is None:
                continue
            cci = data["three_aspect_summary"]
            cci_embedding = data["three_aspect_summary_embedding"]
            query_result = collection.query(
                query_embeddings=cci_embedding,
                n_results=1,
                where={"cve": {"$nin": [data["cve"]]}}
            )
            history_cci = query_result["documents"][0][0]
            history_cve_id = query_result["metadatas"][0][0]["cve"]
            try:
                history_cve_description = cve_ds[cve_index[history_cve_id]]["desc"]
            except KeyError:
                continue

            yield (data["cve"], data["commit_id"], data["diff"], cci, history_cci, history_cve_description, args)
    

    with ThreadPool(args.num_threads) as pool, ThreadPool(args.num_threads) as pool2, ThreadPool(args.num_threads) as pool3:
        cci_results = pool.imap_unordered(process_cci, task_args_gen(), chunksize=1)
        embedding_results = pool2.imap_unordered(process_embedding, task_cci_args_gen(cci_results), chunksize=1)
        cavfd_results = pool3.imap_unordered(process_cavfd, task_rag_args_gen(embedding_results), chunksize=1)
        with open(output_file, "a") as f:
            for result in tqdm(cavfd_results, total=len(resumable_ds), desc=f"Processing {args.subset}", smoothing=0.1):
                if result is None:
                    print("Error processing example, skipping.")
                    continue
                else:
                    try:
                        desc = cve_ds[cve_index[result["cve"]]]["desc"]
                        example = cands_ds[cand_index[result["commit_id"]]]
                    except KeyError:
                        continue
                    
                    # try to json serialize the result, if it fails, skip the example
                    
                    analysis = vulnerability_fix = error = None
                    try:
                        cavfd = json.loads(result["cavfd"])
                        analysis = cavfd.get("analysis")
                        vulnerability_fix = cavfd.get("vulnerability_fix")
                    except json.JSONDecodeError as e:
                        error = f"JSONDecodeError: {e}"
                        
                    
                    f.write(json.dumps({
                        "input": {
                            "cve": example["cve"],
                            "desc": desc,
                            "repo": example["repo"],
                            "commit_id": example["commit_id"],
                            "commit_message": example["commit_message"],
                            "diff": example["diff"],
                            "label": example["label"]
                        },
                        "output": {
                            "analysis": analysis,
                            "vulnerability_fix": vulnerability_fix,
                            "error": error
                        },
                        "metadata": {
                            "description": args.subset,
                            "model": args.model,
                            "embedding_model": args.embedding_model
                        }
                    }) + "\n")
                    f.flush()


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Compute lexical similarity for CVE-commit dataset.")
    #dataset and split args
    parser.add_argument("--output_dir", type=str, default='../../data/baselines/LLM4VFD/output', help="Output directory for lexical similarity CSVs")
    parser.add_argument("--subset", type=str, default="PatchFinder_top10", help="Subset of the dataset to use")
    parser.add_argument("--openai-api-endpoint", type=str, default="http://localhost:8000/v1", help="OpenAI API compatible endpoint to embeddings model")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API key for embeddings model")
    parser.add_argument("--openai-embedding-api-endpoint", type=str, default="http://localhost:8000/v1", help="OpenAI API compatible endpoint to embeddings model")
    parser.add_argument("--openai-embedding-api-key", type=str, default="", help="OpenAI API key for embeddings model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Model that is used for inference")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Model that was used for embeddings")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for concurrent processing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)