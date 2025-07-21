from smolagents import Tool

class ScrollFileTool(Tool):
    name = "scroll_file"
    description = """
    A tool to scroll up or down through the currently opened file. Scrolls by one window size.
    """
    inputs = {
        "direction": {
            "type": "string",
            "description": "The direction to scroll: 'up' or 'down'."
        }
    }
    output_type = "string"

    def __init__(self, windowed_file):
        super().__init__(self)
        self.windowed_file = windowed_file


    def forward(self, direction: str):
        return self.windowed_file.scroll(direction)
