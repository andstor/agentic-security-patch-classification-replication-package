
# %%
from multiprocessing import Queue, Process, BoundedSemaphore, get_context
import polars as pl
from functools import partial
from tqdm import tqdm
import os
import pandas as pd
from bert_score import score, plot_example



mp_context = get_context('spawn')

# %%
DATA_DIR = '../../../data/baselines/PatchFinder'

# %%
from pathlib import Path
from typing import Generator, Dict


# %%
def compute_similarity(df, device_id, batch_size):
    import polars as pl
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Fill NaNs
    df["desc_token"] = df["desc_token"].fillna("")
    df["msg_token"] = df["msg_token"].fillna("")
    df["diff_token"] = df["diff_token"].fillna("")

    # Build combined field
    df["combined"] = df["msg_token"] + " " + df["diff_token"]
    
    
    similarity_scores = score(cands=df['combined'].to_list(), refs=df['desc_token'].to_list(),
                              model_type='microsoft/codereviewer', batch_size=batch_size, device=f"cuda:{device_id}")

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = df['cve']
    similarity_data['repo'] = df['repo']
    similarity_data['commit_id'] = df['commit_id']
    similarity_data['label'] = df['label']
    similarity_data['recall'] = similarity_scores[0].tolist()
    similarity_data['precision'] = similarity_scores[1].tolist()
    similarity_data['f1'] = similarity_scores[2].tolist()

    return similarity_data
    # Append to CSV
    #similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_data.csv'), mode='a', header=False, index=False)



# %%

def constrained_iterator(sem: BoundedSemaphore, data: iter):
    for i in data:
        sem.acquire()
        yield i

def progress_tracker_worker(progressq: Queue):
    from tqdm import tqdm
    #total = 0
    #ds_cve = load_dataset('fals3/cvevc_cve')
    #for split in ds_cve:
    #    total += len(ds_cve[split])
    
    pbar = tqdm(desc="Processed CVE groups", total=10874, dynamic_ncols=True)
    while True:
        item = progressq.get()
        if item is None:
            break
        pbar.update(1)
        
# %%
# Worker to write data into different shard files
def csv_writer_worker(writeq: Queue):
    while True:
        try:
            item = writeq.get()
            if item is None:
                break

            try:
                # Get data and its respective file from the queue
                data, file_suffix = item
                # Write data to the respective shard CSV file
                file_path = os.path.join(DATA_DIR, f'semantic_similarity_{file_suffix}.csv')
                data.to_csv(file_path, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error writing data: {e}")
        except queue.Empty:
            continue  

def owns(cve, worker_id, num_workers):
    return hash(cve) % num_workers == worker_id

def data_producer(dfq: Queue, sem: BoundedSemaphore, worker_id: int, num_workers: int):
    import queue
    from datasets import load_from_disk, load_dataset

    ds_cve = load_from_disk("tmp/ds_cve")
    ds_patches = load_from_disk("tmp/ds_patches")
    ds_nonpatches = load_from_disk("tmp/ds_nonpatches")
    ds_mappings = load_dataset("fals3/cvevc_cve_commit_mappings")

    for split in ["train", "test", "validation"]:

        # Index for commit_id lookup
        cindex = {key: idx for idx, key in tqdm(enumerate(ds_cve[split]["cve"]), total=len(ds_cve[split]["cve"]), desc=f"Indexing CVEs {split}")}
        pindex = {key: idx for idx, key in tqdm(enumerate(ds_patches[split]["commit_id"]), total=len(ds_patches[split]["commit_id"]), desc=f"Indexing patch commits {split}")}
        npindex = {key: idx for idx, key in tqdm(enumerate(ds_nonpatches[split]["commit_id"]), total=len(ds_nonpatches[split]["commit_id"]), desc=f"Indexing non-patch commits {split}")}
        cve_mappings = ds_mappings[split].to_pandas().groupby("cve")
        
        
        def process_mapping(example):
            cve = example["cve"]
            commit_id = example["commit_id"]
            label = example["label"] 
            
            cve_row = ds_cve[split][cindex[cve]]
            
            commit_row = None
            if commit_id in pindex: # Patch commit
                commit_row = ds_patches[split][pindex[commit_id]]
            else: # Non-patch commit
                commit_row = ds_nonpatches[split][npindex[commit_id]]
            
            if commit_row is not None:
                return {
                    "cve": cve,
                    "repo": commit_row["repo"],
                    "commit_id": commit_id,
                    "label": label,
                    "desc_token": cve_row["desc_token"],
                    "msg_token": commit_row["msg_token"],
                    "diff_token": commit_row["diff_token"]
                }
            else:
                return None

        for cve, group_df in constrained_iterator(sem, cve_mappings):
            if not owns(cve, worker_id, num_workers): # Load balancing
                sem.release()
                continue
            try:
                cve_df = group_df.apply(process_mapping, axis=1, result_type='expand')
                cve_df.dropna(inplace=True)
                if cve_df.empty:
                    sem.release()
                    continue
                dfq.put((cve_df, split))
            except Exception as e:
                print(f"Error processing group DataFrame (data_producer): {e}")
                sem.release()
                continue

# %%
import polars as pl

def similarity_worker(dfq: Queue, writeq: Queue, progressq: Queue, sem: BoundedSemaphore, device_id: int, batch_size: int):   
    import queue

    while True:
        try:
            item = dfq.get()
            if item is None:
                break
            df, split = item
            try:
                result = compute_similarity(df, device_id, batch_size)
                writeq.put((result, split))
                progressq.put(1)  # signal 1 group done

            except Exception as e:
                print(f"Error processing data (similarity_worker): {e}")
                continue
            finally:
                sem.release()
        except queue.Empty:
            continue

def main(args):

    
    for split in ["train", "test", "validation"]:
        # Create and write the header of the CSV file
        empty_df = pd.DataFrame(columns=['cve', 'repo', 'commit_id', 'label', 'recall', 'precision', 'f1'])
        empty_df.to_csv(os.path.join(args.data_dir, f'semantic_similarity_{split}.csv'), index=False)

    progressq = mp_context.Queue()
    progress_tracker = mp_context.Process(target=progress_tracker_worker, args=(progressq,))
    progress_tracker.start()
    # %%
    # Create a separate process for writing to the CSV file

    # Number of shards (write processes)
    writeq = mp_context.Queue()
    csv_writer = mp_context.Process(target=csv_writer_worker, args=(writeq, args.data_dir))
    csv_writer.start()

    # %%


    sem = mp_context.BoundedSemaphore(args.max_cves) # the peak number of processed groups.
    dfq = mp_context.Queue()

    data_producer_workers = [
        mp_context.Process(target=data_producer, args=(dfq, sem, i, args.loading_jobs), daemon=True)
        for i in range(args.loading_jobs)
    ]
    for p in data_producer_workers:
        p.start()

    # %%

    device_ids = list(range(args.processing_jobs))

    similarity_workers = []
    for device_id in device_ids:
        p = mp_context.Process(
            target=similarity_worker,
            args=(dfq, writeq, progressq, sem, device_id, args.batch_size),
            daemon=True
        )
        similarity_workers.append(p)
    for p in similarity_workers:
        p.start()


    # %%
    for p in data_producer_workers:
        p.join()

    for _ in range(args.processing_jobs):
        dfq.put(None)
    for p in similarity_workers:
        p.join()

    


    progressq.put(None)
    progress_tracker.join()

    writeq.put(None)
    csv_writer.join()



import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Compute semantic similarity for CVE-commit dataset.")
    parser.add_argument("--data_dir", type=str, default='../../../data/baselines/PatchFinder', help="Output directory for semantic similarity CSVs")
    parser.add_argument("--loading_jobs", type=int, default=6, help="Number of data loading processes")
    parser.add_argument("--processing_jobs", type=int, default=1, help="Number of similarity worker (GPU) processes")
    parser.add_argument("--max_cves", type=int, default=50, help="Maximum number of preloaded CVEs in memory")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for BERTScore computation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
