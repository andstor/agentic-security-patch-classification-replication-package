import os
from src.environment import Environment


class WindowedFileOld:
    """
    Handles the logic for opening and viewing files in a paginated manner.
    """
    def __init__(self, base_path, window_size=100):
        self.base_path = base_path
        self.window_size = window_size
        self.window_start = 0
        self.file_lines = []
        self.environment = Environment()
        self.last_opened_file = None  # Keep track of the last opened file

    def open_file(self, path: str):
        # Construct the absolute file path and read the file
        file_path = os.path.join(self.base_path, path)

        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

        # If reopening the same file, retain the previous window_start
        if file_path == self.last_opened_file:
            # Keep the old window_start position
            pass
        else:
            return f"Error: File {file_path} is already opened."
            #self.window_start = 0  # Reset to the start for a new file

        with open(file_path, "r") as file:
            self.file_lines = file.readlines()

        self.environment.set_current_file(file_path)
        self.last_opened_file = file_path  # Store the current file as the last opened
        return self.get_current_window()

    def go_to(self, line_number: int):
        if not self.file_lines:
            return "Error: No file is currently opened."

        if line_number < 1 or line_number > len(self.file_lines):
            return f"Invalid line number. Must be between 1 and {len(self.file_lines)}."

        self.window_start = line_number - 1
        return self.get_current_window()

    def scroll(self, direction: str):
        if not self.file_lines:
            return "Error: No file is currently opened."

        if direction == "up":
            self.window_start = max(0, self.window_start - self.window_size)
        elif direction == "down":
            self.window_start = min(len(self.file_lines) - self.window_size, self.window_start + self.window_size)
        else:
            return "Invalid direction. Use 'up' or 'down'."

        return self.get_current_window()

    def get_current_window(self):
        if not self.file_lines:
            return "Error: No file is currently opened."

        end = min(self.window_start + self.window_size, len(self.file_lines))
        window_content = self.file_lines[self.window_start:end]
        window_content_str = "".join(window_content)
        return f"Viewing lines {self.window_start + 1} to {end}: of {len(self.file_lines)}\n" + window_content_str