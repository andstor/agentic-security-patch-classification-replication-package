import os
import sys
import json
import argparse
import logging
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from smolagents import OpenAIServerModel
import requests

import time, requests

import time, requests

def healthcheck(url, retries=5, delay=2):
    for _ in range(retries):
        try:
            requests.get(url, timeout=5).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            time.sleep(delay)
    return False

from src.patch_classifier import PatchClassifier

# Set up logging at the module level
logger = logging.getLogger(__name__)


import pickle
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run patch classification agent on CVE candidates.")
    parser.add_argument("--dataset", type=str, default="fals3/cvevc_candidates", help="Dataset to use for CVE candidates")
    parser.add_argument("--subset", type=str, default="PatchFinder_top10", help="Subset of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Model to use for inference")
    parser.add_argument("--openai-api-endpoint", type=str, default="http://localhost:8000/v1", help="OpenAI API compatible endpoint")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API key")
    parser.add_argument("--trace", action="store_true", help="Enable tracing of spans")
    parser.add_argument("--collector-endpoint", type=str, default="http://localhost:4318/v1/traces", help="Endpoint for the tracing collector")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory to store results")
    parser.add_argument("--local-dir", type=str, default="tmp", help="Directory to store cloned repos")
    parser.add_argument("--log-file", type=str, default="application.log", help="Path to the log file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

    args = parser.parse_args()
    load_dotenv(".env")

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_dir, exist_ok=True)


    model = OpenAIServerModel(
        api_base=args.openai_api_endpoint,#"https://vllm1.andstor.dev/v1",
        api_key=args.openai_api_key,
        model_id=args.model,
    #    max_completion_tokens=5000
    )
    
    health_url = args.openai_api_endpoint.replace("/v1", "/health")
    
    project_name = args.subset + "_" + model.model_id.split("/")[-1]
    logger.info(f"Project name: {project_name}")

    if args.trace:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from openinference.semconv.resource import ResourceAttributes
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        from src.tracing import TraceSpanTracker, trace_id_to_hex

        #endpoint = "http://host.docker.internal:6006/v1/traces"
        endpoint = args.collector_endpoint
        resource = Resource(attributes={ResourceAttributes.PROJECT_NAME: project_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        #tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        tracker = TraceSpanTracker()
        tracer_provider.add_span_processor(tracker)
        SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
    else:
        logger.warning("🚨 Tracing is disabled. No spans will be recorded.")

    # Load the dataset
    dataset = load_dataset(args.dataset, args.subset, split=args.split)

    output_file = os.path.join(args.output_dir, f"{project_name}.jsonl")

    if not args.overwrite and os.path.exists(output_file): # Resume from existing data
        logger.info(f"⏭️  Output file {output_file} already exists. Resuming from existing data.")
        processed_ids = set()
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add((data["input"]["cve"], data["input"]["commit_id"]))

        resumable_dataset = dataset.filter(
            lambda example: (example["cve"], example["commit_id"]) not in processed_ids
        )
    else:
        resumable_dataset = dataset
    
    
    classifier = PatchClassifier(
        model=model,
        max_steps=20,  # Set the maximum steps for the agent
        return_full_result=True,
        local_dir=args.local_dir,
        log_file=args.log_file
    )
    #classifier.visualize()

    # Process each example in the dataset
    for example in tqdm(resumable_dataset, smoothing=0.1):
        
        if not healthcheck(health_url, retries=5, delay=3):
            print("❌ Server is unhealthy. Aborting experiment.")
            sys.exit(1) 

        repo_url = f"https://github.com/{example['repo']}.git"
        try:
            output, full_result = classifier.predict(example["cve"], repo_url, example["commit_id"], example["diff"])
        except Exception as e:
            logger.error(f"Error processing commit {example['commit_id']} for CVE {example['cve']}: {e}")
            continue
        data = {
            "input": example,
            "output": output,
            "metadata": {
                "description": args.subset,
                "model": model.model_id,
            }
        }
        
        if args.trace:
            traceid, _ = tracker.get_last_trace()
            trace_id = trace_id_to_hex(traceid) if traceid is not None else None
            data["metadata"]["trace_id"] = trace_id
            tracker.clear()  # Clear tracker for the next example

        # Save the output to a file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(data, f)
            f.write("\n")
            
        # Also save the full result as a pickle file inside a tar archive
        import pickle
        import tarfile
        import io
        
        tarfile_path = os.path.join(args.output_dir, f"{project_name}.tar")
        filename = f"{example['cve']}_{example['commit_id']}.pkl"

        with tarfile.open(tarfile_path, "a") as tar:
            data = pickle.dumps(full_result)
            fileobj = io.BytesIO(data)
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(tarinfo=info, fileobj=fileobj)



if __name__ == "__main__":
    main()