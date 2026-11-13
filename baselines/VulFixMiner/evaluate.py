# %%
import os
import sys
from datasets import load_dataset
import torch
import pandas as pd
from tqdm import tqdm
from model import VulFixMinerFineTuneClassifier
from transformers import RobertaTokenizer
from utils import extract_file_diffs, get_code_version
from model import VulFixMinerClassifier
from torch.nn import functional as F


# Get the absolute path of the parent directory
module_path = os.path.abspath(os.path.join('..'))
# Add the parent directory to sys.path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'mps' if torch.mps.is_available() else device


def main(args):
    cands_ds = load_dataset("fals3/cvevc_candidates", args.subset)
    output_csv_filename = f"predictions_candidates_{args.subset}.csv"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'mps' if torch.mps.is_available() else device

    # ## Load Fine-tuned Model
    finetune_model = VulFixMinerFineTuneClassifier()
    finetune_model.load_state_dict(torch.load("./output/finetuned_model.pt", map_location=torch.device('cpu')))

    if hasattr(finetune_model, "module"):
        code_bert = finetune_model.module.code_bert
    else:
        code_bert = finetune_model.code_bert
        
    code_bert.to(device)
    code_bert.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # ### Tokenization

    ds_diffs = cands_ds.map(
            lambda x: {"diff": extract_file_diffs(x["diff"]).values()},
            batched=False,
            batch_size=1,
            num_proc=10,
            desc="Extracting file diffs from commits"
    )
    
    
    def process_added_removed(row):
        file_codes = []
        for diff in row["diff"]:
            
            added_code = get_code_version(diff=diff, added_version=True)
            deleted_code = get_code_version(diff=diff, added_version=False)
            
            if not added_code and not deleted_code: # if all comments or empty lines
                continue
            file_codes.append(added_code + tokenizer.sep_token + deleted_code)
        row["code"] = file_codes
        
        return row
        
    ds_code = ds_diffs.map(
        process_added_removed,
        batched=False,
        batch_size=1,
        num_proc=10,
        remove_columns=["diff"],
        desc="Processing added and removed changes"
    )
    
    ds_code = ds_code.filter(
        lambda x: len(x["code"]) > 0,
        batched=False,
        num_proc=10
    )
    
    
    def preprocess_function(examples):
        res = tokenizer(examples["code"], add_special_tokens=True, max_length=256, truncation=True, padding="max_length")
        res["label"] = examples["label"]
        return res

    ds_tokenized = ds_code.map(preprocess_function, batched=False, num_proc=10, desc="Tokenizing")#, remove_columns=ds_code["train"].column_names)
    ds_tokenized = ds_tokenized.with_format("torch")
    
    
    def get_embeddings(row):
        try:
            with torch.no_grad():
                inputs = row['input_ids'].to(device)
                attention = row['attention_mask'].to(device)
                outputs = code_bert(inputs, attention)
                embeddings = outputs.last_hidden_state[:, 0, :]
                mean_ = torch.mean(embeddings, dim=0).cpu()
            row["embedding"] = mean_
        except torch.OutOfMemoryError:
            print(f"⚠️ Skipping sample (too large): {len(row['input_ids'])} tokens")
            row = None
            torch.cuda.empty_cache()  # free up memory to continue
        return row

    predictions = ds_tokenized.map(
        get_embeddings,
        batched=False,
        num_proc=1,
        desc="Computing embeddings",
    )

    model = VulFixMinerClassifier()
    model.load_state_dict(torch.load("./output/trained_model.pt", map_location=torch.device('cpu')))
    model.eval()
    model.to(device)


    ds = predictions.with_format("torch", columns=["embedding"], output_all_columns=True, device=device)

    columns=['cve', 'commit_id', 'label', 'prediction']
    predictions_df = pd.DataFrame(columns=columns)
    # --- Inference over the entire test dataset and append to CSV with progress tracking ---
    output_filepath = os.path.join(args.output_dir, output_csv_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_df.to_csv(output_filepath, index=False, mode='w', header=True) # Create empty CSV with header


    for example in tqdm(ds["test"]):
        input = example['embedding'].unsqueeze(0)
        with torch.no_grad():
            output = model(input)
        probabilities = F.softmax(output, dim=1)[:, 1].tolist()
        
        example['prediction'] = probabilities[0]
        df = pd.DataFrame([example])
        df = df[columns]
        df.to_csv(output_filepath, index=False, mode='a', header=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CVE Patch Classifier")
    parser.add_argument('--output-dir', type=str, default='../../data/baselines/VulFixMiner', help="Directory to save the output CSV files")
    parser.add_argument('--subset', type=str, required=True, help="Subset of dataset to evaluate on (e.g., 'PatchFinder_top10', 'random_10')")
    args = parser.parse_args()

    main(args)


