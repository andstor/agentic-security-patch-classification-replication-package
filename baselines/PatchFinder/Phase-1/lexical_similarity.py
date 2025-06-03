
# %%
from multiprocessing import Queue, Process, BoundedSemaphore, get_context
import polars as pl
from functools import partial
from tqdm import tqdm
import os
import pandas as pd

mp_context = get_context('spawn')

# %%
DATA_DIR = '../../../data/baselines/PatchFinder'

# %%
from pathlib import Path
from typing import Generator, Dict

def get_parquet_files_by_split(base_dir: Path = Path("tmp/owner_repo_groups")) -> Dict[str, Generator[Path, None, None]]:
    """Return a dict mapping 'train', 'test', 'validation' to generators of .parquet file paths."""
    splits = ["train", "test", "validation"]
    return {
        split: sorted((base_dir / split).glob("*.parquet"))
        for split in splits
    }

# %%
def compute_similarity(df):
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
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['combined'])

    similarity_scores = []
    for row in df.iter_rows(named=True):
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['combined']])
        similarity = cosine_similarity(desc_tfidf, combined_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = df['cve']
    similarity_data['owner'] = df['owner']
    similarity_data['repo'] = df['repo']
    similarity_data['commit_id'] = df['commit_id']
    similarity_data['similarity'] = similarity_scores
    similarity_data['label'] = df['label']

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
    #for split in ["train", "test", "validation"]:
    #    cve_path = f".tmp/tokenized/cve_{split}.parquet"
    #    count = pl.scan_parquet(cve_path).collect().shape[0]
    #    total += count
    
    pbar = tqdm(desc="Processed CVE groups", total=11943, dynamic_ncols=True)
    while True:
        item = progressq.get()
        if item is None:
            break
        pbar.update(1)
        
# %%
# Worker to write data into different shard files
def csv_writer_worker(writeq: Queue):
    import queue
    while True:
        try:
            item = writeq.get()
            if item is None:
                break

            try:
                # Get data and its respective file from the queue
                data, file_suffix = item
                # Write data to the respective shard CSV file
                file_path = os.path.join(DATA_DIR, f'lexical_similarity_{file_suffix}.csv')
                data.to_csv(file_path, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error writing data: {e}")
        except queue.Empty:
            continue  
            

def data_producer(fileq: Queue, dfq: Queue, sem: BoundedSemaphore):
    import queue

    while True:
        try:
            item = fileq.get()
            if item is None:
                break
            split, file = item
            try:
                cve_path = f"tmp/tokenized/cve_{split}.parquet"
                patches_path = f"tmp/tokenized/patches_{split}.parquet"

                dsp_cve = pl.scan_parquet(cve_path)
                dsp_patches = pl.scan_parquet(patches_path)
                dsp_nonpatches = pl.scan_parquet(file)
            
                
                dsp_cve_exploded = dsp_cve.explode("commits").rename({"commits": "commit_id"})
                dsp_patches = ( # tmp join with patches to get owner, repo cols
                    dsp_cve_exploded
                    .join(dsp_patches, on="commit_id", how="inner")
                )
                dsp_cve = dsp_patches.select(["cve", "owner", "repo", "desc_token"]).unique(pl.col("cve")).collect()
                dsp_patches = dsp_patches.select(['cve', 'owner', 'repo', 'commit_id', 'label', 'desc_token', 'msg_token', 'diff_token']).collect()
                
                
                
                ####
                nonpatch_owner_repo = dsp_nonpatches.select(["owner", "repo"]).unique()
                dsp_patches_filtered = (
                    dsp_patches.lazy()
                    .join(nonpatch_owner_repo, on=["owner", "repo"], how="inner")
                    .select(["cve", "owner", "repo", "commit_id", "label", "desc_token", "msg_token", "diff_token"])
                )
                
                
                dsp_nonpatches_joined = ( # join with non-patch data
                    dsp_cve.lazy()
                    .join(dsp_nonpatches, on=["owner", "repo"], how="right")
                    .select (['cve', 'owner', 'repo', 'commit_id', 'label', 'desc_token', 'msg_token', 'diff_token'])
                )
                
                dsp_commits_lazy = pl.concat([dsp_patches_filtered, dsp_nonpatches_joined], how="vertical")
                grouped = dsp_commits_lazy.collect().group_by("cve")
                for _, group_df in constrained_iterator(sem, grouped):
                    dfq.put((group_df, split))
            except Exception as e:
                print(f"Error processing data (data_producer): {e}")
                continue
        except queue.Empty:
            continue    

# %%
import polars as pl

def similarity_worker(dfq: Queue, writeq: Queue, progressq: Queue, sem: BoundedSemaphore):   
    import queue

    while True:
        try:
            item = dfq.get()
            if item is None:
                break
            df, split = item
            try:
                result = compute_similarity(df)
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

    # Example usage
    files_by_split = get_parquet_files_by_split()
    for split, files in files_by_split.items():
        print(f"{split}: {len(files)} files")


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


    LOADING_JOBS = 11 # MAX amount of files in memory
    MAX_CVES = 50

    sem = mp_context.BoundedSemaphore(MAX_CVES) # the peak number of processed groups.
    fileq = mp_context.Queue() # commits data
    dfq = mp_context.Queue()

    data_producer_workers = [
        mp_context.Process(target=data_producer, args=(fileq, dfq, sem), daemon=True)
        for _ in range(LOADING_JOBS)
    ]
    for p in data_producer_workers:
        p.start()

    # %%

    PROCESSING_JOBS = 11

    similarity_workers = [
        mp_context.Process(target=similarity_worker, args=(dfq, writeq, progressq, sem), daemon=True)
        for _ in range(PROCESSING_JOBS)
    ]
    for p in similarity_workers:
        p.start()

    # %%

    files_by_split = get_parquet_files_by_split()
    for split in files_by_split.keys():

        # Create and write the header of the CSV file
        empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
        empty_df.to_csv(os.path.join(DATA_DIR, f'lexical_similarity_{split}.csv'), index=False)
            
        
        for file in files_by_split[split]:
            fileq.put((split, file))

    # %%
    for _ in range(LOADING_JOBS):
        fileq.put(None)
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
