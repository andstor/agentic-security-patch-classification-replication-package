import uuid
from git import Repo
from smolagents import CodeAgent, LogLevel
from typing import Any
from jinja2 import Template, StrictUndefined
import yaml
from rich.console import Console
import os
import json
import importlib
import yaml

from src.tools.windowed_file import WindowedFile
from src.tools.code_search_tool import CodeSearchTool
from src.tools.file_search_tool import FileSearchTool
from src.tools.open_file_tool import OpenFileTool
from src.tools.scroll_file_tool import ScrollFileTool
from src.tools.cve_report_tool import CVEReportTool
from src.tools.cwe_report_tool import CWEReportTool
from src.git import clone_repo
from src.utils import delete_subfolder_safely



class PatchClassifier():
    """
    A class to classify patches based on CVE relevance using a CodeAgent.
    """

    def __init__(self, model, max_steps, local_dir, return_full_result=False, log_file=None):
        self.model = model
        self.max_steps = max_steps
        self.return_full_result = return_full_result
        self.local_dir = local_dir
        self.log_file = log_file

        with importlib.resources.open_text("src", "prompts.yaml") as f:
            self.prompt_template = yaml.safe_load(f)

    def predict(self, cve_id, repo_url, commit_id, commit_diff=None):
        """
        Make a prediction about the commit.
        """
        file = None
        try:

            repo_path = os.path.join(self.local_dir, uuid.uuid4().hex)
            repo = clone_repo(repo_url, commit_id, repo_path=repo_path)
            target_commit = repo.commit(commit_id)
            # Get its parent (use [0] to get the first parent if merge)
            if target_commit.parents:
                parent_commit = target_commit.parents[0]
                repo = clone_repo(repo_url, parent_commit.hexsha, repo_path=repo_path)
            else:
                raise ValueError(f"Commit {commit_id} has no parent")
            
            if commit_diff is None:
                commit_diff = self.get_diff(repo, target_commit)


            question = self.populate_template(
                template=self.prompt_template["task_prompt"],
                variables={
                    "cve_id": cve_id,
                    "commit_id": commit_id,
                    "repository": repo_url,
                    "commit_diff": commit_diff,
                }
            )
            agent = self.setup_agent(repo)
            if self.log_file:
                file = open(self.log_file, "wt")
                agent.logger.console = Console(file=file, force_jupyter=False, width=160)
            
            result = answer = confidence = explanation = error = None
            try:
                result = agent.run(task=question)
                output = result.output
                if isinstance(output, str):
                    # Parse the answer if it's a string
                    output = json.loads(output)
                    
                answer = output.get("answer")
                explanation = output.get("explanation")
                confidence = output.get("confidence")
                
                # check confidence is a number between 1 and 5. else set to None
                if not isinstance(confidence, int) or confidence < 1 or confidence > 5:
                    confidence = None

                assert answer in [True, False], "Answer must be either True or False"
                assert isinstance(output.get("confidence"), int) and 1 <= output.get("confidence") <= 5, "Confidence must be an integer between 1 and 5"
                assert isinstance(explanation, str) and len(explanation) > 0, "Explanation must be a non-empty string"
            except json.JSONDecodeError as e:
                error = f"JSONDecodeError: {e}"
            except AssertionError as e:
                error = str(e)
            
        finally:
            if file:
                file.close()
                
            # Clean up the cloned repository
            try:
                folder_to_delete = os.path.dirname(repo_path)
                delete_subfolder_safely(folder_to_delete, self.local_dir)
            except Exception as e:
                pass
        
        return_result = {
            "answer": answer,
            "confidence": confidence,
            "explanation": explanation,
            "error": error
        }
        
        if self.return_full_result:
            print("A")
            return return_result, result
        else:
            print("B")
            return return_result

    def populate_template(self, template: str, variables: dict[str, Any]) -> str:
        compiled_template = Template(template, undefined=StrictUndefined)
        try:
            return compiled_template.render(**variables)
        except Exception as e:
            raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

    def get_diff(self, repo, commit):
        raise NotImplementedError("This method should be implemented to extract the diff from the commit.")

    def setup_agent(self, repo):
        """
        Set up the CodeAgent with the repository and model.
        """
        windowed_file = WindowedFile(base_path=repo.working_dir)
        agent = CodeAgent(
            tools=[
                CVEReportTool(),
                CWEReportTool(),
                FileSearchTool(repo_path=repo.working_dir),
                CodeSearchTool(repo_path=repo.working_dir),
                OpenFileTool(windowed_file),
                ScrollFileTool(windowed_file)
            ],
            model=self.model,
            return_full_result=True,
            verbosity_level=LogLevel.DEBUG,
            max_steps=self.max_steps,
            instructions=""
        )
        return agent
    
    def visualize(self):
        """
        Visualize the agent setup.
        """
        # make temporary repo 
        tmp_repo = Repo.init(os.path.join(self.local_dir, "tmp_repo"))
        agent = self.setup_agent(tmp_repo)
        agent.visualize()
