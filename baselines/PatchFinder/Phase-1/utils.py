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
