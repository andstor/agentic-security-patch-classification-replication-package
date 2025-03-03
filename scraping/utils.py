from git import RemoteProgress
from tqdm import tqdm

class CloneProgress(RemoteProgress):
    def __init__(self, kwargs):
        super().__init__()
        self.pbar = tqdm(leave=False, position=kwargs["position"])

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        
        # Map operation code to operation name.
        op_code_to_name = {
            self.COUNTING: "counting objects",
            self.COMPRESSING: "compressing objects",
            self.WRITING: "writing objects",
            self.RECEIVING: "receiving objects",
            self.RESOLVING: "resolving deltas",
            self.FINDING_SOURCES: "finding sources",
            self.CHECKING_OUT: "checking out files"
        }

        # Get operation name.
        op_name = op_code_to_name.get(op_code, "")
        self.pbar.set_description(f"Cloning ({op_name}) {message}")
        self.pbar.refresh()



import random

def pick_random_commits(repo, num_commits, seed=None):
    """
    Pick a specified number of random commits from a Git repository using reservoir sampling.

    :param repo: A GitPython Repo object
    :param num_commits: The number of random commits to pick
    :param seed: An optional seed for the random number generator (default: None)
    :return: A list of randomly selected commits
    """
    if seed is not None:
        random.seed(seed)  # Set the random seed for reproducibility

    if num_commits <= 0:
        raise ValueError("The number of commits to pick must be greater than 0.")
    
    # Initialize reservoir
    reservoir = []
    count = 0

    # Iterate over all commits lazily
    for count, commit in enumerate(repo.iter_commits(), 1):  # Start counting at 1
        if len(reservoir) < num_commits:
            # Fill the reservoir initially
            reservoir.append(commit)
        else:
            # Replace elements with gradually decreasing probability
            replace_idx = random.randint(0, count - 1)
            if replace_idx < num_commits:
                reservoir[replace_idx] = commit

    if count < num_commits:
        raise ValueError(f"Requested {num_commits} commits, but the repository only has {count} commits.")

    return reservoir


def get_commit_diff(commit):
    """
    Get the changes (diff) for a given commit compared to its parent.

    :param commit: A GitPython Commit object
    :return: A string representation of the diff
    """
    if not commit.parents:
        # Handle the case of the first commit (no parent)
        return f"Commit {commit.hexsha} is the root commit. No diff available."

    # Get the diff compared to the parent
    parent = commit.parents[0]
    diff = parent.diff(commit, create_patch=True)

    # Generate a unified diff string
    diff_strings = []
    for change in diff:
        diff_strings.append(change.diff.decode('utf-8', errors='ignore'))

    return "\n".join(diff_strings)


def get_commit_diff_custom_format(commit):
    """
    Get the diff for a given commit compared to its parent, formatted in the desired diff format.

    :param commit: A GitPython Commit object
    :return: A string with the formatted diff output
    """
    if not commit.parents:
        # Handle the case of the first commit (no parent)
        return f"Commit {commit.hexsha} is the root commit. No diff available."

    # Get the parent commit (assume single-parent commit)
    parent = commit.parents[0]
    diff = parent.diff(commit, create_patch=True)

    diff_output = []
    for change in diff:
        # Get the hash for the commit before and after
        old_commit_hash = parent.hexsha
        new_commit_hash = commit.hexsha

        # Get the file mode for old and new versions
        old_mode = change.a_mode if hasattr(change, 'a_mode') else 'unknown'
        new_mode = change.b_mode if hasattr(change, 'b_mode') else 'unknown'

        # Format the diff output similar to the Git diff format
        diff_output.append(f"diff --git a/{change.b_path} b/{change.b_path}")
        diff_output.append(f"index {old_commit_hash[:7]}..{new_commit_hash[:7]} {new_mode}")
        diff_output.append(f"--- a/{change.b_path}")
        diff_output.append(f"+++ b/{change.b_path}")

        # Adding the unified diff (lines added and removed)
        diff_output.append(change.diff.decode('utf-8', errors='ignore'))

        # Handle any additional metadata, like new or deleted file
        if change.new_file:
            diff_output.append(f"new file mode {new_mode}")
        if change.deleted_file:
            diff_output.append(f"deleted file mode {old_mode}")
        if change.renamed_file:
            diff_output.append(f"rename from {change.a_path}")
            diff_output.append(f"rename to {change.b_path}")

    return "\n".join(diff_output)


def get_commit_info(commit):
    output = []
    output.append(f"Author: {commit.author}")
    output.append(f"Date: {commit.authored_datetime}")
    output.append("")
    output.append(f"{commit.message}")

    return "\n".join(output)


import re

def extract_commit_hash(input_string):
    """
    Extract the commit hash from any string.

    Args:
        input_string (str): The input string to search for a commit hash.

    Returns:
        str or None: The first valid commit hash if found, otherwise None.
    """
    pattern = r"\b[a-fA-F0-9]{40}\b"  # Regex for a 40-character hexadecimal hash
    match = re.search(pattern, input_string)
    return match.group(0) if match else None


import re
from typing import Optional, Tuple

def extract_github_repo(url: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the repository owner and name from a GitHub URL.

    Args:
        url (str): The GitHub URL.

    Returns:
        Optional[Tuple[str, str]]: A tuple containing (owner, repo) if found, otherwise None.
    """
    pattern = r"https?://github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)

    if match:
        return match.groups()
    return None




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


import itertools

def take(iterable, n):
    li = list(itertools.islice(iterable, n))
    return li