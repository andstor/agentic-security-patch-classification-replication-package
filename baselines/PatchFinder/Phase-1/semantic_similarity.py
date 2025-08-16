
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
def compute_similarity(df, device_id):
    import polars as pl
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = df.with_columns(
        pl.col("desc_token").fill_null(''),
        pl.col("msg_token").fill_null(''),
        pl.col("diff_token").fill_null(''),
    ).with_columns(
        pl.concat_str([pl.col("msg_token"), pl.col("diff_token")], separator=" ").fill_null('').alias("combined")
    )
    
    
    similarity_scores = score(cands=df['combined'].to_list(), refs=df['desc_token'].to_list(),
                              model_type='microsoft/codereviewer', batch_size=1024, device=f"cuda:{device_id}")

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
    
    pbar = tqdm(desc="Processed CVE groups", total=11936, dynamic_ncols=True)
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
            

def data_producer(dfq: Queue, sem: BoundedSemaphore):
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

        for _, group_df in constrained_iterator(sem, cve_mappings):
            try:
                cve_df = group_df.apply(process_mapping, axis=1)
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

def similarity_worker(dfq: Queue, writeq: Queue, progressq: Queue, sem: BoundedSemaphore, device_id: int):   
    import queue

    while True:
        try:
            item = dfq.get()
            if item is None:
                break
            df, split = item
            try:
                result = compute_similarity(df, device_id)
                writeq.put((result, split))
                progressq.put(1)  # signal 1 group done

            except Exception as e:
                print(f"Error processing data (similarity_worker): {e}")
                continue
            finally:
                sem.release()
        except queue.Empty:
            continue

def main():

    
    for split in ["train", "test", "validation"]:
        # Create and write the header of the CSV file
        empty_df = pd.DataFrame(columns=['cve', 'repo', 'commit_id', 'label', 'recall', 'precision', 'f1'])
        empty_df.to_csv(os.path.join(DATA_DIR, f'semantic_similarity_{split}.csv'), index=False)
            

    progressq = mp_context.Queue()
    progress_tracker = mp_context.Process(target=progress_tracker_worker, args=(progressq,))
    progress_tracker.start()
    # %%
    # Create a separate process for writing to the CSV file

    # Number of shards (write processes)
    writeq = mp_context.Queue()
    csv_writer = mp_context.Process(target=csv_writer_worker, args=(writeq,))
    csv_writer.start()

    # %%


    LOADING_JOBS = 4 # MAX amount of files in memory
    MAX_CVES = 50

    sem = mp_context.BoundedSemaphore(MAX_CVES) # the peak number of processed groups.
    dfq = mp_context.Queue()

    data_producer_workers = [
        mp_context.Process(target=data_producer, args=(dfq, sem), daemon=True)
        for _ in range(LOADING_JOBS)
    ]
    for p in data_producer_workers:
        p.start()

    # %%

    PROCESSING_JOBS = 8 # Num GPUs
    device_ids = list(range(PROCESSING_JOBS))

    similarity_workers = []
    for device_id in device_ids:
        p = mp_context.Process(
            target=similarity_worker,
            args=(dfq, writeq, progressq, sem, device_id),
            daemon=True
        )
        similarity_workers.append(p)
    for p in similarity_workers:
        p.start()

    # %%

    # %%
    for p in data_producer_workers:
        p.join()

    for _ in range(PROCESSING_JOBS):
        dfq.put(None)
    for p in similarity_workers:
        p.join()

    


    progressq.put(None)
    progress_tracker.join()

    writeq.put(None)
    csv_writer.join()


if __name__ == "__main__":
    main()
