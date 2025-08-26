from git import RemoteProgress
from tqdm import tqdm
import os
import random
import re
import shutil  # Import shutil for deleting directories

class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

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







def delete_subfolder_safely(folder_to_delete, expected_parent_folder):
    """
    Safely deletes a folder, ensuring it is a subfolder of a specified parent folder.

    Args:
        folder_to_delete (str): The path to the folder to be deleted.
        expected_parent_folder (str): The path to the expected parent folder.

    Returns:
        bool: True if deletion succeeded, False otherwise.
    """
    abs_folder_to_delete = os.path.realpath(folder_to_delete)
    abs_expected_parent_folder = os.path.realpath(expected_parent_folder)

    if os.path.commonpath([abs_folder_to_delete, abs_expected_parent_folder]) == abs_expected_parent_folder:
        try:
            shutil.rmtree(abs_folder_to_delete)
            #print(f"Successfully deleted: {abs_folder_to_delete}")
            return True
        except Exception as e:
            #print(f"Error deleting {abs_folder_to_delete}: {e}")
            return False
    else:
        #print(f"Error: {abs_folder_to_delete} is not a subfolder of {abs_expected_parent_folder}. Deletion aborted.")
        return False

