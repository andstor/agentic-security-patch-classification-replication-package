from datasets import load_dataset
from utils import process_patch
from openai import OpenAI
from prompts import SYSTEM_PROMPT_CAVFD, SYSTEM_PROMPT_CCI, SYSTEM_PROMPT_DA, USER_PROMPT_CAVFD, USER_PROMPT_CCI, USER_PROMPT_DA
import os
from tqdm import tqdm
import json
from multiprocessing.dummy import Pool as ThreadPool
import chromadb

def process_example(args_tuple):
    """Worker function to process a single example"""
    cve, commit_id, desc, three_aspect_summary, args = args_tuple

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
            "documents": three_aspect_summary,
            "embedding": embedding
        }
    except Exception as e:
        return None

def main(args):
    
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    print(chroma_client.list_collections())
    
    model_name = args.model.split("/")[1]
    collection = None
    try:
        collection = chroma_client.get_collection(name=f"3aspect_summary_{model_name}")
    except Exception as e:
        collection = chroma_client.create_collection(
            name=f"3aspect_summary_{model_name}",
            metadata={"model": args.model, "embedding_model": args.embedding_model}
        )

    cve_ds = load_dataset("fals3/cvevc_cve")
    mappings_ds = load_dataset("fals3/cvevc_cve_commit_mappings")
    mappings_ds = mappings_ds.filter(lambda x: x["label"] == 1, num_proc=10) # keep only positive samples


    for split in ["train", "validation", "test"]:
        input_file = os.path.join(args.input_dir, f"{model_name}_{split}.jsonl")
        
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        # Prepare a lookup for cve descriptions
        cindex = {key: idx for idx, key in tqdm(enumerate(cve_ds[split]["cve"]), total=len(cve_ds[split]["cve"]), desc=f"Indexing CVEs {split}")}

        # Prepare arguments for thread pool as a generator
        def task_args_gen():
            for example in data:
                try:
                    cve = cve_ds[split][cindex[example["cve"]]]
                    yield (example["cve"], example["commit_id"], cve["desc"], example["three_aspect_summary"], args)
                except KeyError:
                    continue

        with ThreadPool(args.num_threads) as pool:
            results = pool.imap_unordered(process_example, task_args_gen())
            for result in tqdm(results, total=len(data), desc=f"Processing {split}"):
                if result is None:
                    print("Error processing example, skipping.")
                    continue
                else:
                    collection.add(
                        documents=result["documents"],
                        embeddings=result["embedding"],
                        metadatas={"cve": result["cve"], "commit_id": result["commit_id"], "split": split},
                        ids=result["commit_id"],
                    )


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Compute lexical similarity for CVE-commit dataset.")
    parser.add_argument("--input_dir", type=str, default='../../data/baselines/LLM4VFD/3aspect_summaries', help="Input directory for lexical similarity CSVs")
    parser.add_argument("--openai-embedding-api-endpoint", type=str, default="http://localhost:8000/v1", help="OpenAI API compatible endpoint to embeddings model")
    parser.add_argument("--openai-embedding-api-key", type=str, default="", help="OpenAI API key for embeddings model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Model that was used for inference")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Model to use for embeddings")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for concurrent processing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)