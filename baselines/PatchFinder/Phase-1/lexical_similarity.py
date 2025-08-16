
# %%
from multiprocessing import Queue, Process, BoundedSemaphore, get_context
import polars as pl
from functools import partial
from tqdm import tqdm
import os
import pandas as pd
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
import polars as pl
import pandas as pd
import re




mp_context = get_context('spawn')

# %%
DATA_DIR = '../../../data/baselines/PatchFinder'

# %%
from pathlib import Path
from typing import Generator, Dict


import nltk
nltk.download('punkt_tab')

from nltk import word_tokenize




import re

def convert_to_unified_0(diff: str) -> str:
    """
    Takes a git diff string and returns a version equivalent to `git diff --unified=0`.
    """
    output_lines = []
    diff_lines = diff.splitlines()
    
    inside_diff = False
    
    for line in diff_lines:
        if line.startswith("diff --git") or line.startswith("index") or line.startswith("---") or line.startswith("+++"):
            output_lines.append(line)
        elif line.startswith("@@"):
            inside_diff = True
            # Extract hunk header and modify it to show 0 lines of context
            match = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
            if match:
                old_start, old_count, new_start, new_count = match.groups()
                old_count = int(old_count) if old_count else 1
                new_count = int(new_count) if new_count else 1
                output_lines.append(f"@@ -{old_start},0 +{new_start},0 @@")
            else:
                output_lines.append(line)
        elif inside_diff:
            if line.startswith("+") or line.startswith("-"):
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    return "\n".join(output_lines)







import re

def format_git_show_minimal(git_show_string):
    """
    Robustly extracts diff content starting from the first '@@' line for each file, including the 'diff --git' line.

    Args:
        git_show_string: The git show diff string with potentially multiple file diffs.

    Returns:
        The extracted diff content, or an empty string if no diff is found.
    """
    lines = git_show_string.splitlines()
    result_diffs = []
    current_diff = []
    at_at_found = False

    for line in lines:
        if line.startswith("diff --git"):
            if current_diff:  # Store the previous diff if any
                result_diffs.append("\n".join(current_diff))
            current_diff = [line]  # Start a new diff
            at_at_found = False
        elif current_diff:
            if line.startswith("@@"):
                at_at_found = True
                current_diff.append(line)
            elif at_at_found:
                current_diff.append(line)

    if current_diff:  # Store the last diff
        result_diffs.append("\n".join(current_diff))

    return "\n".join(result_diffs).strip()





def load_data(num_cpus=20):
    import os
    import json
    import pandas as pd
    from datasets import load_dataset

    # Load CVE dataset
    ds_cve = load_dataset('fals3/cvevc_cve')
    ds_cve = ds_cve.map(lambda x: {"desc_token": ' '.join(word_tokenize(x["desc"]))}, batched=False, num_proc=num_cpus)

    # Load patches dataset
    ds_patches = load_dataset('fals3/cvevc_commits', "patches")
    ds_patches = ds_patches.map(lambda x: {"diff_token": 
                                               ' '.join(word_tokenize(
                                                   ''.join(format_git_show_minimal(
                                                       convert_to_unified_0(
                                                           x["diff"]
                                                       )
                                                   ).splitlines(keepends=True)[:1000])
                                               )),
                                           "msg_token": ' '.join(word_tokenize(x["commit_message"]))
                                          }, batched=False, num_proc=num_cpus)
    ds_patches = ds_patches.remove_columns(["commit_message", "diff"])

    # Load non-patches dataset
    ds_nonpatches = load_dataset("fals3/cvevc_commits", "non_patches")
    ds_nonpatches = ds_nonpatches.map(lambda x: {"diff_token": 
                                                     ' '.join(word_tokenize(
                                                         "".join(format_git_show_minimal(
                                                             convert_to_unified_0(
                                                                 x["diff"]
                                                             )
                                                         ).splitlines(keepends=True)[:1000])
                                                     )),
                                                 "msg_token": ' '.join(word_tokenize(x["commit_message"]))
                                                }, batched=False, num_proc=num_cpus)
    ds_nonpatches = ds_nonpatches.remove_columns(["commit_message", "diff"])
    
    # Load CVE to commit mappings
    mapping_ds = load_dataset("fals3/cvevc_cve_commit_mappings", num_proc=num_cpus)
    
    return ds_cve, ds_patches, ds_nonpatches, mapping_ds





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
            

def data_producer(dfq: Queue, sem: BoundedSemaphore):
    import queue

    ds_cve, ds_patches, ds_nonpatches, ds_mappings = load_data()
    
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
    
    
    for split in ["train", "test", "validation"]:
        # Create and write the header of the CSV file
        empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
        empty_df.to_csv(os.path.join(DATA_DIR, f'lexical_similarity_{split}.csv'), index=False)
            
    load_data() # Preload data to ensure all datasets are available. Subsequent calls to load_data() are cached.

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
    dfq = mp_context.Queue()

    data_producer_workers = [
        mp_context.Process(target=data_producer, args=(dfq, sem), daemon=True)
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
