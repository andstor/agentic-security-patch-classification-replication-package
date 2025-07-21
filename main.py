import os
import sys
import json
import pickle
import argparse
from datasets import load_dataset
import polars as pl
from git import Repo
from dotenv import load_dotenv

from smolagents import OpenAIServerModel, CodeAgent
from src.prompts import USER_PROMPT_PATCH_INFO, USER_PROMPT_TIER_3
from src.tools.windowed_file import WindowedFile
from src.tools.code_search_tool import CodeSearchTool
from src.tools.file_search_tool import FileSearchTool
from src.tools.open_file_tool import OpenFileTool
from src.tools.scroll_file_tool import ScrollFileTool
from src.tools.cve_info_tool import CVEReportTool


def process_single_cve(cve_id, cands_df, model, local_dir, output_dir, overwrite=False):
    # Directory for all results related to this CVE (main results and traces)
    cve_output_dir = os.path.join(output_dir, cve_id)
    os.makedirs(cve_output_dir, exist_ok=True)

    print(f"\n🔍 Processing CVE: {cve_id}")
    cands = cands_df.filter(pl.col("cve") == cve_id)

    if len(cands) == 0:
        print(f"⚠️ No candidates found for {cve_id}")
        return

    repo_name = cands["repo"].first()
    user = cands["owner"].first()
    repo_url = f"https://github.com/{user}/{repo_name}.git"
    repo_path = os.path.join(local_dir, repo_name)


    if not os.path.exists(repo_path):
        print(f"📥 Cloning {repo_url}")
        try:
            Repo.clone_from(repo_url, repo_path, depth=1)
        except Exception as e:
            print(f"❌ Failed to clone repo: {e}")
            return

    repo = Repo(repo_path)

    windowed_file = WindowedFile(base_path=repo_path)
    cve_report_tool = CVEReportTool()

    agent = CodeAgent(
        tools=[
            cve_report_tool,
            CodeSearchTool(repo_path=repo_path),
            FileSearchTool(repo_path=repo_path),
            OpenFileTool(windowed_file),
            ScrollFileTool(windowed_file),
        ],
        model=model,
        verbosity_level=2,
        max_steps=20,
        planning_interval=3,
        name="security_patch_identification_agent",
    )

    for row in cands.iter_rows(named=True):
        commit_id = row["commit_id"]
        print(f"→ Commit {commit_id}")

        # Paths for the result and trace files for this specific commit
        commit_results_json_path = os.path.join(cve_output_dir, f"{commit_id}_results.json")
        commit_trace_json_path = os.path.join(cve_output_dir, "trace", f"{commit_id}_trace.json")
        commit_trace_pickle_path = os.path.join(cve_output_dir, "trace", f"{commit_id}_trace.pkl")
        # Ensure trace directory exists
        os.makedirs(os.path.dirname(commit_trace_json_path), exist_ok=True)

        if not overwrite and os.path.exists(commit_results_json_path):
            print(f"⏭️  Skipping commit {commit_id} (results file already exists and overwrite is false)")
            continue

        try:
            repo.git.fetch("origin", commit_id)
            repo.git.checkout("FETCH_HEAD")
            repo.git.reset("--hard")
        except Exception as e:
            print(f"❌ Git error on {commit_id}: {e}")
            continue

        try:
            task = USER_PROMPT_TIER_3.format(
                user_prompt_patch_info=USER_PROMPT_PATCH_INFO.format(
                    cve_id=cve_id,
                    commit_id=commit_id,
                    repository=f"{row['owner']}/{row['repo']}",
                    commit_message=row["commit_message"],
                    commit_diff=row["diff"]
                )
            )
            answer = agent.run(task=task)
            trace = agent.memory.get_full_steps()
        except Exception as e:
            print(f"⚠️  Agent failed on {commit_id}: {e}")
            continue

        predicted_label = "true" in str(answer).lower()

        # Main results dictionary including trace for the combined JSON file
        result = {
            "cve_id": cve_id,
            "commit_id": commit_id,
            "ground_truth": True if row["label"] == 1 else False,
            "prediction": predicted_label,
        }

        # Save the combined results (including trace) to a JSON file
        with open(commit_results_json_path, "w") as f:
            f.write(json.dumps(result, indent=2))

        # Save the trace separately to a JSON file
        with open(commit_trace_json_path, "w") as f:
            f.write(json.dumps(trace, default=str, indent=2))

        # Save the trace separately to a Pickle file
        with open(commit_trace_pickle_path, "wb") as f:
            f.write(pickle.dumps(trace))

        print(f"✅ Saved results to {commit_results_json_path}, trace to {commit_trace_json_path}, {commit_trace_pickle_path}")

def main():
    parser = argparse.ArgumentParser(description="Run patch classification agent on CVE candidates.")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory to store results")
    parser.add_argument("--local-dir", type=str, default=".tmp", help="Directory to store cloned repos")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of CVEs to process")
    parser.add_argument("--cve", type=str, help="Process only a single CVE (e.g. CVE-2019-10782)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()
    load_dotenv(".env")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_dir, exist_ok=True)

    # Load dataset
    cands_ds = load_dataset('fals3/cvevc_candidates', split='test')
    cands_ds = cands_ds.filter(lambda x: x['rank'] <= 10)
    cands_df = cands_ds.to_polars().sort(['cve', 'commit_id'])

    # Model setup
    model = OpenAIServerModel(
        api_base="https://holy-quickly-flea.ngrok-free.app/v1",
        api_key=os.getenv("API_KEY", ""),
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=10000,
    )

    if args.cve:
        process_single_cve(args.cve, cands_df, model, args.local_dir, args.output_dir, args.overwrite)
    else:
        cve_list = (
            cands_df.filter(pl.col("label") == 1)
            .select(pl.col("cve"))
            .unique()
            .to_series()
            .to_list()
        )
        if args.limit:
            cve_list = cve_list[:args.limit]

        print(f"Processing {len(cve_list)} CVEs.")
        for cve_id in cve_list:
            process_single_cve(cve_id, cands_df, model, args.local_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()