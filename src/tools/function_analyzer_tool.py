from smolagents import Tool, CodeAgent, HfApiModel
from typing import Optional
import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
)
from src.tools.grep_tool import GrepTool
from src.tools.scroll_file_tool import ScrollFileTool
from src.tools.open_file_tool import OpenFileTool
from src.tools.file_search_tool import FileSearchTool
from src.tools.windowed_file import WindowedFile
from src.tools.goto import GotoTool



class FunctionAnalyzer(Tool):
    name = "function_call_analyzer"
    description = """
    A advanced tool to analyze a function in a file. Provides a detailed report on the function.
    """
    inputs = {
        "function_name": {
            "type": "string",
            "description": "The name of the function to analyze."
        },
        "path": {
            "type": "string",
            "description": "The path to the file the function is defined in."
        },
        "line_number": {
            "type": "integer",
            "description": "The line number where the function is defined. Optional.",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, repo_path, parent=None):
        super().__init__()

        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.model = HfApiModel(model_id, token=os.getenv("HF_TOKEN"))
        
        self.repo_path = repo_path
        self.parent = parent
        self.windowed_file = WindowedFile(base_path="")
        

    def forward(self, function_name: str, path: str, line_number: Optional[int] = None):
        self.last_function_name = function_name
        # Uses agents to analyze the function
        if self.parent:
            print("parent", self.parent.last_function_name)
            if self.parent.last_function_name == function_name:
                return "Error: You are already analyzing this function."
        
        
        
        #WeakMultiStepAgent
        agent = CodeAgent(
            tools=[FunctionAnalyzer(repo_path=self.repo_path, parent=self), OpenFileTool(windowed_file=self.windowed_file), ScrollFileTool(windowed_file=self.windowed_file), GotoTool(windowed_file=self.windowed_file)],
            #tools=[FileSearchTool(repo_path=repo.working_dir), GrepTool(repo_path=repo.working_dir)],
            model=self.model,
            #managed_agents=[definition_descriptor_agent],
            max_steps=10,
            verbosity_level=2,
        )
        agent.system_prompt += ""
        
        question = """What does the function `{function_name}` in the file `{path}`{at_line} do?. Provide a exhaustive description of the function. First find the code. Then analyze it. You may need to understand what some function calls do (beyond its name). In that case, you can locate where the function defenition is, and then use the FunctionAnalyzer tool. It will not work to use it on the {function_name}!!!!! Begin!""".format(
            function_name=function_name,
            path=path,
            at_line=f" at line {line_number}" if line_number else ""
        )
        
        result = agent.run(question, additional_args={"repo_path": self.repo_path})
        return result
