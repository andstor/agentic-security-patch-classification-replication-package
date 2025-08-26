from smolagents import Tool
from git.exc import GitCommandError
from typing import Optional

class CodeSearchTool(Tool):
    name = "code_search"
    description = "Tool for searching for file contents in a git repository files."
    inputs = {
        "query": {
            "type": "string",
            "description": "The string to search for. Can only be simple string."
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
            res = self.repo.git.grep("-n", "-F", query, file)

            # If more than 50 files are found, return a message asking to be more specific
            if len(res.split("\n")) > 100:
                return "Too many results found. Please be more specific."
            
            #res = res.split("\n")
            #res = [line.split(":") for line in res if line]
            #res = [{"file": line[0], "line": line[1], "content": ":".join(line[2:])} for line in res]
            
            # If matches are found, return the res list
            return res
        except GitCommandError as e:
            # Handle the case where no matches are found
            return f"No matches found for '{query}'."
        except Exception as e:
            # Handle other exceptions
            return f"An error occurred while searching: {str(e)}"