import multiprocessing as mp
import queue
from multiprocessing import Queue, Process, BoundedSemaphore

import random
import time
import pickle
from tqdm import tqdm
from utils import CloneProgress, extract_commit_hash
import os
from git import Repo
import shutil
import json
import threading
from utils import extract_github_repo, take
random.seed(42)

from pathlib import Path
LOCAL_DIR = "./.tmp"
OUTPUT_DIR = "./output"

JOBS = 10
CONCURRENT_DOWNLOADS = 10

RETAINED_REPOS = 10
RETAINED_REPOS = max(CONCURRENT_DOWNLOADS, RETAINED_REPOS)


def git_clone_worker(repo_ident):
    pos = mp.current_process()._identity[0] + JOBS
    
    repo_owner, repo_name = repo_ident.split("/")
    
    repo_path = Path(LOCAL_DIR) / (repo_owner + ":" + repo_name)
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    try:
        if os.path.exists(repo_path):
            Repo(repo_path)
        else:
            Repo.clone_from(repo_url, Path(LOCAL_DIR) / (repo_owner + ":" + repo_name), progress=CloneProgress(kwargs={"position": pos, "dynamic_ncols": True}))

        return repo_path
    
    except Exception as e:
        return repo_path

def constrained_iterator(sem: threading.BoundedSemaphore, data: iter):
    for i in data:
        sem.acquire()
        yield i

def jsonl_writer_worker(q: Queue):
    while True:
        try:
            m = q.get()
            if m is None:
                break
            data, file = m
            with open(Path(OUTPUT_DIR) / file, "a") as f:
                f.write(json.dumps(data))
                f.write("\n")
                f.flush()
        except queue.Empty:
            continue

def main_worker(mainq: Queue, writeq: Queue, sem: BoundedSemaphore): # type: ignore
    pos = mp.current_process()._identity[0]
    
    while True:
        try:
            item = mainq.get()
            if item is None:
                break
            repo_path, cves = item
    
            try:
                repo = Repo(repo_path)
                repo_owner, repo_name = repo_path.parts[-1].split(":")
                
                with tqdm(position=pos, leave=False, desc=f"Extracting commits from {repo_owner}/{repo_name}", dynamic_ncols=True) as pbar:
                    patch_commits = []
                    for cve_id, cve in cves.items():
                        for commit_url in cve["ground_truth"]["commit"]:
                            patch_ref = extract_commit_hash(commit_url)
                            if patch_ref is None:
                                continue
                            try:
                                commit = repo.commit(patch_ref)
                            except ValueError:
                                continue
                            
                            patch_commits.append(commit)
                            diff = commit.repo.git.show(commit.hexsha) # Non-plumbing commands in git output in utf-8.
                            
                            #cve,owner,repo,commit_id,label,desc_token,msg_token,diff_token
                            data = {
                                "commit_id": commit.hexsha,
                                "owner": repo_owner,
                                "repo": repo_name,
                                "commit_message": commit.message,
                                "diff": diff,
                            }
                            writeq.put((data, "patches_data.jsonl"))
                            pbar.update(1)
                    
                    # get latest 20000 commits
                    commits = take(repo.iter_commits(), 20000)
                    random.shuffle(commits)
                    #select 5000 commits or all commits if less than 5000
                    count = 0
                    while count < 5000 and commits:
                        commit = commits.pop()
                        if commit in patch_commits:
                            continue
                        diff = commit.repo.git.show(commit.hexsha)
                        data = {
                            "commit_id": commit.hexsha,
                            "owner": repo_owner,
                            "repo": repo_name,
                            "commit_message": commit.message,
                            "diff": diff,
                        }
                        writeq.put((data, "commits_data.jsonl"))
                        count += 1
                        pbar.update(1)
            
            except Exception:
                continue
            
            finally:
                try:
                    repo_path = Path(LOCAL_DIR) / (repo_owner + ":" + repo_name)
                    if os.path.exists(repo_path):
                        shutil.rmtree(repo_path, ignore_errors=True)
                except BaseException as e:
                    pass
                finally:
                    sem.release()
        except queue.Empty:
            continue

def take_named_entries(d, keys):
    return {k: d[k] for k in keys if k in d}

def main():
    with open('../data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    grouped_dataset = {}
    for cve_id, cve in tqdm(dataset.items(), desc="Grouping dataset"):
        commit_url = cve["ground_truth"]["commit"][0]
        github_url = extract_github_repo(commit_url)
        if github_url:
            owner, repo = github_url
            ident = owner + "/" + repo
            grouped_dataset.setdefault(ident, []).append(cve_id)

    # Create a separate process for writing to the JSONL file
    writeq = Queue()
    jsonl_writer = mp.Process(target=jsonl_writer_worker, args=(writeq,))
    jsonl_writer.start()
    
    
    sem = mp.BoundedSemaphore(RETAINED_REPOS)

    mainq = Queue() # commits data
    main_workers = [
        Process(target=main_worker, args=(mainq, writeq, sem), daemon=True)
        for _ in range(JOBS)
    ]
    for p in main_workers:
        p.start()

    
    with mp.Pool(processes=CONCURRENT_DOWNLOADS) as pool:
        repos = grouped_dataset.keys()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with tqdm(total=len(grouped_dataset), position=0, desc="Scraping repositories", dynamic_ncols=True) as pbar:
            for repo_path in pool.imap_unordered(git_clone_worker, constrained_iterator(sem, repos)):
                
                try:
                    repo = Repo(repo_path)
                    repo_owner, repo_name = repo_path.parts[-1].split(":")
                    repo_ident = repo_owner + "/" + repo_name
                    meta = {
                        "owner": repo_owner,
                        "repo": repo_name,
                        "head": repo.head.commit.hexsha,
                    }
                    with open(Path(OUTPUT_DIR) / "meta.json", "a") as f:
                        f.write(json.dumps(meta))
                        f.write("\n")
                    
                    cve_ids = grouped_dataset[repo_ident]
                    cves = take_named_entries(dataset, cve_ids)
                    mainq.put((repo_path, cves)) # Process the cves
                
                except Exception as e:
                    print(f"Error main processing {repo_path}: {e}")
                    continue
                
                finally:
                    pbar.update(1)
    
    for _ in range(JOBS):
        mainq.put(None)
    for p in main_workers:
        p.join()
    
    
    writeq.put(None)
    jsonl_writer.join()

if __name__ == "__main__":
    main()
