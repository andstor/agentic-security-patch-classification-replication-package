import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchmetrics.classification import BinaryAccuracy  # Also supports multi-GPU
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
from load_data import CVEDataset  # Import the dataset class
from model import CVEClassifier  # Adjust import path if needed

# --- Configuration ---
data_path = '../../../data/baselines/PatchFinder'
input_csv_filename = "hybrid_similarity_test.csv"  # Name of your input CSV file
output_csv_filename = "predictions_test.csv" # Name for the output CSV file
batch_size = 32
num_workers = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Explicitly set primary CUDA device

# --- Define the CVEClassifier Model (as provided) ---
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pytorch_lightning as pl
import torch.nn as nn
import torch




def main():

    # --- Load the trained model and move parameters to the primary device ---
    try:
        model = torch.load("./output/all/Checkpoints/final_model.pt", weights_only=False) # Load to the specified device
        #model = CVEClassifier.load_from_checkpoint("../output/all/Checkpoints/final_model.ckpt", map_location=device) # If using checkpoint
    except FileNotFoundError:
        print("Error: Trained model file not found. Please check the path.")
        exit()

    model.to(device)
        
    model.eval()

    # --- Load the test dataset and monkeypatch it to return all columns ---
    test_dataset = CVEDataset(os.path.join(data_path, input_csv_filename))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # --- Create an empty DataFrame for storing results ---
    columns=['cve', 'commit_id', 'label', 'prediction']
    predictions_df = pd.DataFrame(columns=columns)

    # --- Inference over the entire test dataset and append to CSV with progress tracking ---
    output_filepath = os.path.join(data_path, output_csv_filename)
    predictions_df.to_csv(output_filepath, index=False, mode='w', header=True) # Create empty CSV with header

    for batch in tqdm(test_dataloader, desc="Processing Batches"):
        # Move input tensors to the correct device
        input_keys = ['input_ids_desc', 'attention_mask_desc', 'input_ids_msg_diff', 'attention_mask_msg_diff']
        inputs = {key: batch[key].to(device) for key in input_keys}
        labels = batch['label'].cpu().numpy().flatten().tolist()
        
        # Inference
        with torch.no_grad():
            output = model(**inputs)
            probabilities = torch.sigmoid(output).cpu().numpy().flatten().tolist()

        # Convert tensor and list fields to CPU/numpy
        batch_data = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cpu().numpy()
            else:
                batch_data[key] = value  # Keep strings, lists, etc.
    
        # Remove tokenized input keys
        for key in input_keys:
            batch_data.pop(key, None)
    
        # Convert label to int
        batch_data['label'] = batch_data['label'].astype(int)

        batch_df = pd.DataFrame(batch_data)
        batch_df['prediction'] = probabilities
        
        # Reorder columns to match the desired output structure

        batch_df = batch_df[columns]
        
        # Append the current batch's predictions to the CSV file
        batch_df.to_csv(output_filepath, index=False, mode='a', header=False)


    print(f"\nPredictions saved to: {output_filepath}")

    # --- Optionally, merge with the original DataFrame if needed ---
    #original_df = pd.read_csv(os.path.join(data_path, input_csv_filename))
    #final_df = pd.merge(original_df, pd.read_csv(output_filepath), on='cve', how='inner')
    #final_df.to_csv("merged_" + output_csv_filename, index=False)
    #print(f"\nMerged predictions with original data saved to: merged_{output_csv_filename}")


if __name__ == '__main__':
    main()