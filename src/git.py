import uuid
from git import Repo
import os

def clone_repo(repo_url, commit_id, repo_path):

    #check if the repo is already cloned
    if os.path.exists(repo_path):
        repo = Repo(repo_path)
        #logger.info(f"Using existing repo at {repo_path}")
    else:
        #logger.info(f"Cloning {repo_url} into {repo_path}")
        repo = Repo.init(repo_path)
        repo.create_remote("origin", repo_url)

    repo.git.fetch("--depth", "1", "origin", commit_id)
    repo.git.checkout("FETCH_HEAD")
    
    return repo



import re
def extract_diff(git_show_string):

    if not git_show_string:
        return {}
    lines = git_show_string.splitlines()
    diff = []
    inside_diff = False

    for line in lines:
        if line.startswith("diff --git"):
            inside_diff = True
        if inside_diff:
            diff.append(line)

    return "\n".join(diff)