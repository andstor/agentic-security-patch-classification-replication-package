import multiprocessing as mp
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
import random
random.seed(42)

from pathlib import Path
LOCAL_DIR = "./.tmp"
OUTPUT_DIR = "./output"



def git_clone_worker(repo_ident):
    pos = mp.current_process()._identity[0]
    
    repo_owner, repo_name = repo_ident.split("/")
    
    repo_path = Path(LOCAL_DIR) / (repo_owner + ":" + repo_name)
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    try:
        if os.path.exists(repo_path):
            Repo(repo_path)
        else:
            Repo.clone_from(repo_url, Path(LOCAL_DIR) / (repo_owner + ":" + repo_name), progress=CloneProgress(kwargs={"position": pos}))

        return repo_path
    
    except Exception as e:
        return repo_path



def constrained_iterator(sem: threading.BoundedSemaphore, data: iter):
    for i in data:
        sem.acquire()
        yield i


def main():
    
    
    with open('../data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    grouped_dataset = {}
    for cve_id, cve in tqdm(dataset.items(), desc="Grouping dataset"):
        commit_url = cve["ground_truth"]["commit"][0] # only first commit for now
        github_url = extract_github_repo(commit_url)
        if github_url:
            owner, repo = github_url
            ident = owner + "/" + repo
            if ident not in grouped_dataset:
                grouped_dataset[ident] = []
                grouped_dataset[ident].append(cve_id)
            else:
                grouped_dataset[ident].append(cve_id)

    num_proc = 5
    
    sem = threading.BoundedSemaphore(num_proc)
    with mp.Pool(processes=num_proc) as pool:
        # skip linux
        repos = (grouped_dataset.keys())
        
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stats = {"failure": 0, "commits": 0}
        
        with tqdm(total=len(grouped_dataset), position=0) as pbar:
            for repo_path in pool.imap_unordered(git_clone_worker, constrained_iterator(sem, repos)):
                sem.release()
                
                try:
                    repo = Repo(repo_path)
                    repo_owner, repo_name = repo_path.parts[-1].split(":")
                    repo_ident = repo_owner + "/" + repo_name
                    pbar.set_description(repo_ident)
                    meta = {
                        "owner": repo_owner,
                        "repo": repo_name,
                        "head": repo.head.commit.hexsha,
                    }
                    with open(Path(OUTPUT_DIR) / "meta.json", "a") as f:
                        f.write(json.dumps(meta))
                        f.write("\n")
                        
                        
                        
                        
                        
                    patch_commits = []
                    with open(Path(OUTPUT_DIR) / "patches_data.jsonl", "a") as f:
                        for cve_id in grouped_dataset[repo_ident]:
                            cve = dataset[cve_id]
                            # save patch
                            
                            for commit_url in cve["ground_truth"]["commit"]:
                                patch_ref = extract_commit_hash(commit_url)
                                pbar.set_description(f"Processing {cve_id} ({commit_url})")
                                if patch_ref is None:
                                    continue
                                try:
                                    commit = repo.commit(patch_ref) # b2ab395adba5d85f7d20ec25ac815add26c1296c
                                except ValueError as e:
                                    continue
                                    
                                patch_commits.append(commit)
                                diff = commit.repo.git.show(commit.hexsha)
                                
                                #cve,owner,repo,commit_id,label,desc_token,msg_token,diff_token
                                data = {
                                    "commit_id": commit.hexsha,
                                    "owner": repo_owner,
                                    "repo": repo_name,
                                    "commit_message": commit.message,
                                    "diff": diff,
                                }
                                
                                f.write(json.dumps(data))
                                f.write("\n")
                                stats["commits"] += 1
                                pbar.set_postfix(stats)
                
                
                
                    # get latest 20000 commits
                    commits = take(repo.iter_commits(), 20000)
                    random.shuffle(commits)
                    #select 5000 commits or all commits if less than 5000
                    count = 0
                    with open(Path(OUTPUT_DIR) / "commits_data.jsonl", "a") as f:
                        while count < 5000:
                            if len(commits) == 0:
                                break
                            commit = commits.pop()
                            if commit in patch_commits:
                                continue
                            
                            diff = commit.repo.git.show(commit.hexsha) # Non-plumbing commands in git output in utf-8.
                            data = {
                                "commit_id": commit.hexsha,
                                "owner": repo_owner,
                                "repo": repo_name,
                                "commit_message": commit.message,
                                "diff": diff,
                            }
                            
                            
                            f.write(json.dumps(data) + "\n")
                            f.flush()

                            count += 1
                            stats["commits"] += 1
                            pbar.set_postfix(stats)
                    
                except Exception as e:
                    stats["failure"] += 1
                    continue
                
                finally:
                    # Cleanup: Remove cloned repository
                    pbar.update(1)
                    try:
                        repo_path = Path(LOCAL_DIR) / (repo_owner + ":" + repo_name)
                        if os.path.exists(repo_path):
                            shutil.rmtree(repo_path, ignore_errors=True)
                    except BaseException:
                        pass

if __name__ == "__main__":
    main()


