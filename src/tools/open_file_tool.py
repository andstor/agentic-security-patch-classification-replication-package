from smolagents import Tool
from typing import Optional
import os


class OpenFileTool(Tool):
    name = "open_file"
    description = """
    A tool to open a file and initialize it for paginated viewing. The window size is set to 100 lines.
    """
    inputs = {
        "path": {
            "type": "string",
            "description": "The path to the file to open."
        },
        #"line_number": {
        #    "type": "integer",
        #    "description": "The line number to start viewing the file from. Optional.",
        #}
    }
    output_type = "string"

    def __init__(self, windowed_file):
        super().__init__()
        self.windowed_file = windowed_file

    def forward(self, path: str):#, line_number: int):
        result = self.windowed_file.open_file(path)
        if "Error" in result:
            return result
        
        #if not line_number:
        #    return "Error: Line number is required."
        
        #if line_number < 1:
        #    return "Error: Line number must be greater than 0."
        
        return result#self.windowed_file.go_to(line_number)
