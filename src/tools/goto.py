from smolagents import Tool

class GotoTool(Tool):
    name = "goto"
    description = """
    A tool to go to a specific line in the currently opened file.
    """
    inputs = {
        "line": {
            "type": "integer",
            "description": "The line number to go to in the file."
        }
    }
    output_type = "string"

    def __init__(self, windowed_file):
        super().__init__(self)
        self.windowed_file = windowed_file


    def forward(self, line: str):
        return self.windowed_file.go_to(line)
