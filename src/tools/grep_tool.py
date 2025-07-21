from smolagents import Tool
from git.exc import GitCommandError
from typing import Optional

class GrepTool(Tool):
    name = "grep"
    description = """
    This is a tool that searches for a string in a git repository.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The string to search for."
        },
        "file": {
            "type": "string",
            "description": "The file to search in. Optional.",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, repo_path):
        super().__init__(self)

        from git import Repo
        self.repo = Repo(repo_path)

    def forward(self, query: str, file: Optional[str] = None):
        try:
            # Perform the search using git grep
            res = self.repo.git.grep("-n", query, file)
            
            # If more than 50 files are found, return a message asking to be more specific
            if len(res.split("\n")) > 100:
                return "Too many files found. Please be more specific."
            
            # If matches are found, return the res list
            return res
        except GitCommandError as e:
            # Handle the case where no matches are found
            return f"No matches found for '{query}'."
        except Exception as e:
            # Handle other exceptions
            return f"An error occurred while searching: {str(e)}"