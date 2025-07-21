
class Environment:
    """
    General singleton class to store the current environment.
    Tracks the currently opened file path or other environment-specific states.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Environment, cls).__new__(cls)
            cls._instance.current_file = None
        return cls._instance

    def set_current_file(self, path):
        self.current_file = path

    def get_current_file(self):
        return self.current_file