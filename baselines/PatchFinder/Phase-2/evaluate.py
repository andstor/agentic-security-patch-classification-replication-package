# %%
import os
import sys
from datasets import load_dataset
from model import CVEClassifier  # Adjust import path if needed
import torch
import pandas as pd
from tqdm import tqdm
# Get the absolute path of the parent directory
module_path = os.path.abspath(os.path.join('..'))
# Add the parent directory to sys.path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)

import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'mps' if torch.mps.is_available() else device


def main(args):
    cands_ds = load_dataset("fals3/cvevc_candidates", args.subset)
    output_csv_filename = f"predictions_candidates_{args.subset}.csv"
    # ### Tokenization
    from Phase_1.utils import convert_to_unified_0, format_git_show_minimal

    def preprocess(example):

        example["desc_token"] = ' '.join(word_tokenize(example["desc"]))
        example["diff_token"] = ' '.join(word_tokenize(
                                        ''.join(format_git_show_minimal(
                                            convert_to_unified_0(
                                            example["diff"]
                                            )
                                    ).splitlines(keepends=True)[:1000])
                                ))
        example["msg_token"] = ' '.join(word_tokenize(example["commit_message"]))
        example["combined"] = example["msg_token"] + " " + example["diff_token"]
        return example
        
        
    prepared_cands_ds = cands_ds.map(preprocess, batched=False, num_proc=4)



    # ### Tokenization
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")

    def tokenize_function(example):
        desc_encoding = tokenizer.encode_plus(
            example["desc_token"],
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        msg_diff_encoding = tokenizer.encode_plus(
            example["combined"],
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids_desc': desc_encoding['input_ids'].flatten(),
            'attention_mask_desc': desc_encoding['attention_mask'].flatten(),
            'input_ids_msg_diff': msg_diff_encoding['input_ids'].flatten(),
            'attention_mask_msg_diff': msg_diff_encoding['attention_mask'].flatten(),
        }

    tokenized_hybrid_cands_ds = prepared_cands_ds.map(tokenize_function, batched=False, num_proc=10)

    # ### Predict
    model = torch.load("./output/Checkpoints/final_model.pt", weights_only=False) # Load to the specified device
    model.to(device)
    model.eval()


    ds = tokenized_hybrid_cands_ds.with_format("torch", columns=[
        'input_ids_desc', 'attention_mask_desc',
        'input_ids_msg_diff', 'attention_mask_msg_diff'
    ], output_all_columns=True, device=device)

    columns=['cve', 'commit_id', 'label', 'prediction']
    predictions_df = pd.DataFrame(columns=columns)
    # --- Inference over the entire test dataset and append to CSV with progress tracking ---
    output_filepath = os.path.join(args.output_dir, output_csv_filename)
    predictions_df.to_csv(output_filepath, index=False, mode='w', header=True) # Create empty CSV with header


    for example in tqdm(ds["test"]):
        input_keys = ['input_ids_desc', 'attention_mask_desc', 'input_ids_msg_diff', 'attention_mask_msg_diff']
        inputs = {key: example[key].unsqueeze(0) for key in input_keys}
        with torch.no_grad():
            output = model(**inputs)
        probabilities = torch.sigmoid(output).cpu().numpy().flatten().tolist()
        
        example['prediction'] = probabilities[0]
        df = pd.DataFrame([example])
        df = df[columns]
        df.to_csv(output_filepath, index=False, mode='a', header=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CVE Patch Classifier")
    parser.add_argument('--output-dir', type=str, default='../../../data/baselines/PatchFinder', help="Directory to save the output CSV files")
    parser.add_argument('--subset', type=str, required=True, help="Subset of dataset to evaluate on (e.g., 'PatchFinder_top10', 'random_10')")
    args = parser.parse_args()

    main(args)


