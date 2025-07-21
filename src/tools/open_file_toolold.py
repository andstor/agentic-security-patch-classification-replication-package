from smolagents import Tool
from typing import Optional

class OpenFileToolOld(Tool):
    name = "open_file"
    description = """
    A tool to open a file and initialize it for paginated viewing. The window size is set to 100 lines.
    """
    inputs = {
        "path": {
            "type": "string",
            "description": "The path to the file to open."
        },
        "line_number": {
            "type": "integer",
            "description": "The line number to start viewing the file from. Optional.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, windowed_file):
        super().__init__(self)
        self.windowed_file = windowed_file

    def forward(self, path: str, line_number: Optional[int] = None):
        if line_number is not None:
            self.windowed_file.open_file(path)
            return self.windowed_file.go_to(line_number)
        return self.windowed_file.open_file(path)