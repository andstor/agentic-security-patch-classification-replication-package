from smolagents import Tool
from git.exc import GitCommandError
import fnmatch

class FileSearchTool(Tool):
    name = "file_search"
    description = "This is a tool that searches for files in a Git repository by filename, keyword, or glob pattern. It uses 'git ls-files' to list files tracked by the repository and supports glob-style searches."
    inputs = {
        "query": {
            "type": "string",
            "description": "The filename, keyword, or glob pattern to search for in the repository."
        }
    }
    output_type = "string"

    def __init__(self, repo_path):
        super().__init__(self)

        from git import Repo
        self.repo = Repo(repo_path)
        
    def forward(self, query: str):
        try:
            # List all tracked files in the repository
            all_files = self.repo.git.ls_files().splitlines()

            # Determine if the query is a glob pattern or a substring
            if any(char in query for char in ['*', '?', '[', ']']):
                # Perform glob-style matching
                matching_files = fnmatch.filter(all_files, query)
            else:
                # Perform substring search (case-insensitive)
                matching_files = [file for file in all_files if query.lower() in file.lower()]

            # Handle results
            if not matching_files:
                return f"No files found matching '{query}'."
            if len(matching_files) > 50:
                return "Too many files found (more than 50). Please refine your search query."

            # Return the matching file paths, one per line
            return "\n".join(matching_files)

        except GitCommandError as e:
            # Handle Git-related errors
            return f"Git error: {str(e)}"
        except Exception as e:
            # Handle general exceptions
            return f"An error occurred: {str(e)}"