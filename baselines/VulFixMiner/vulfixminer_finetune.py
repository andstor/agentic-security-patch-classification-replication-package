from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from torch.optim import AdamW
from transformers import get_scheduler
from patch_entities import VulFixMinerFileDataset
from model import VulFixMinerFineTuneClassifier
from tqdm import tqdm
import pandas as pd
from utils import get_code_version, extract_file_diffs
#import config
import argparse
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

# dataset_name = 'sap_patch_dataset.csv'
# FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'

dataset_name = None
FINE_TUNED_MODEL_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')


NUMBER_OF_EPOCHS = 15
TRAIN_BATCH_SIZE = 64#8
TEST_BATCH_SIZE = 64#32
EARLY_STOPPING_ROUND = 5

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
use_mps = torch.backends.mps.is_available()
device = torch.device("cuda:0" if use_cuda else "mps" if use_mps else "cpu")

random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 256
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.1
NUMBER_OF_LABELS = 2



def predict_test_data(model, testing_generator, device):
    print("Testing...")
    y_pred = []
    y_test = []
    #urls = []
    probs = []
    losses = []
    loss_function = nn.NLLLoss()
    
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(testing_generator, dynamic_ncols=True, leave=True):
            
            input_batch = batch["input_ids"].to(device)
            mask_batch = batch["attention_mask"].to(device)
            label_batch = batch["labels"].to(device)
            
            outs = model(input_batch, mask_batch)
            outs_logsoftmax = F.log_softmax(outs, dim=1)
            outs_softmax = F.softmax(outs, dim=1)
            loss = loss_function(outs_logsoftmax, label_batch)
            losses.append(loss.item())
            y_pred.extend(torch.argmax(outs_softmax, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs_softmax[:, 1].tolist())
            #urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        avg_loss = np.mean(losses) if losses else 0

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    return precision, recall, f1, auc, avg_loss


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    progress_bar = tqdm(range(len(training_generator) * number_of_epochs), desc="", dynamic_ncols=True, leave=True)

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        train_losses = []  # Reset per epoch
        current_batch = 0
        progress_bar.set_description(f"Epoch [{epoch+1}/{number_of_epochs}]")

        for batch in training_generator:
            
            input_batch = batch["input_ids"].to(device)
            mask_batch = batch["attention_mask"].to(device)
            label_batch = batch["labels"].to(device)

            outs = model(input_batch, mask_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1

            avg_loss = np.mean(train_losses)
            progress_bar.update(1)
            progress_bar.set_postfix({
                "batch": current_batch,
                "loss": total_loss,
                "avg_loss": avg_loss
            })

        
        progress_bar.close()

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        print("Result on testing dataset...")
        precision, recall, f1, auc, val_loss = predict_test_data(
            model=model,
            testing_generator=test_generator,
            device=device
        )

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("Validation loss: {}".format(val_loss))
        print("-" * 32)

        if val_loss < best_val_loss: # 
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_ROUND:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

          
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), FINE_TUNED_MODEL_PATH)
    else:
        torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)

    return model


def load_data():
    ds_patches = load_dataset("fals3/cvcvc_commits", "patches")
    ds_patches.pop("validation")  # Remove validation split if it exists, as we will use train/test splits only
    #ddict = DatasetDict()
    #ddict["train"] = Dataset.from_list([x for x in ds_patches["train"].take(10)])
    #ddict["validation"] = Dataset.from_list([x for x in ds_patches["validation"].take(10)])
    #ddict["test"] = Dataset.from_list([x for x in ds_patches["test"].take(10)])
    #ds_patches = ddict
    ds_patches = ds_patches.filter(lambda x: len(x['diff']) <= 45510, batched=False, num_proc=10, desc="Filter out binary files")
    
    ds_nonpatches = load_dataset("fals3/cvcvc_commits", "non_patches")
    ds_nonpatches.pop("validation")  # Remove validation split if it exists, as we will use train/test splits only
    #ddict = DatasetDict()
    #ddict["train"] = Dataset.from_list([x for x in ds_nonpatches["train"].take(10)])
    #ddict["validation"] = Dataset.from_list([x for x in ds_nonpatches["validation"].take(10)])
    #ddict["test"] = Dataset.from_list([x for x in ds_nonpatches["test"].take(10)])
    #ds_nonpatches = ddict
    ds_nonpatches = ds_nonpatches.filter(lambda x: len(x['diff']) <= 45510, batched=False, num_proc=10, desc="Filter out binary files")

    ds_commits = DatasetDict()
    for key in ds_nonpatches:
        ds_commits[key] = concatenate_datasets([ds_nonpatches[key], ds_patches[key]])
    
    
    return ds_commits


def do_train(args):
    global dataset_name, FINE_TUNED_MODEL_PATH

    
    FINE_TUNED_MODEL_PATH = args.finetune_model_path

    print("Saving model to: {}".format(FINE_TUNED_MODEL_PATH))

    
    from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
    #ds_nonpatches = load_dataset("fals3/cvcvc_commits", "patches")
    #ds_nonpatches = ds_patches.filter(lambda x: len(x['diff']) <= 45510, batched=False, num_proc=10)

    #ds_patches = load_dataset("fals3/cvcvc_commits", "patches")
    #ds_patches = ds_patches.filter(lambda x: len(x['diff']) <= 45510, batched=False, num_proc=10)
    
    #ds_commits = DatasetDict()
    #for key in ds_nonpatches:
    #    ds_commits[key] = concatenate_datasets([ds_nonpatches[key], ds_patches[key]])
    
    
    ds_commits = load_data()    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


    

    def explode_file_diffs(batch):
        """
        Splits each commit diff into multiple rows, one per file diff.

        Args:
            batch (dict of lists): A batch from the dataset containing keys like 'diff', 'commit_id', etc.

        Returns:
            dict of lists: Same keys, but expanded so that each file diff is a separate row.
        """
        exploded_data = {}

        # Initialize empty lists for each field
        for key in batch.keys():
            exploded_data[key] = []

        # Add new fields for filename and filediff
        exploded_data['file_path'] = []
        exploded_data['file_diff'] = []

        for i in range(len(batch['diff'])):
            commit_diff = batch['diff'][i]
            file_diffs = extract_file_diffs(commit_diff)
            
            for file_path, file_diff in file_diffs.items():
                # Keep original metadata for each file diff
                for key in batch.keys():
                    exploded_data[key].append(batch[key][i])
                exploded_data['file_path'].append(file_path)
                exploded_data['file_diff'].append(file_diff)

        
        return exploded_data


    ds_files = ds_commits.map(
        explode_file_diffs,
        batched=True,
        batch_size=1,
        num_proc=10,
        desc="Exploding file diffs"
    )

    ds_files = ds_files.remove_columns(["diff"])


    def process_added_removed(row):
        
        added_code = get_code_version(diff=row["file_diff"], added_version=True)
        deleted_code = get_code_version(diff=row["file_diff"], added_version=False)
        
        row["code"] = added_code + tokenizer.sep_token + deleted_code
        
        return row
        
    ds_code = ds_files.map(
        process_added_removed,
        batched=False,
        batch_size=1,
        num_proc=10,
        remove_columns=["file_diff"],
        desc="Processing added and removed changes"
    )


    def preprocess_function(examples):
        res = tokenizer(examples["code"], add_special_tokens=True, max_length=CODE_LENGTH, truncation=True, padding="max_length")
        res["label"] = examples["label"]
        return res

    ds_tokenized = ds_code.map(preprocess_function, batched=False, num_proc=10, remove_columns=ds_code["train"].column_names, desc="Tokenizing")

    ds = ds_tokenized.with_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_generator = DataLoader(
        ds["train"],
        collate_fn=data_collator,
        **TRAIN_PARAMS
    )
    
    test_generator = DataLoader(
        ds["test"],
        collate_fn=data_collator,
        **TEST_PARAMS
    )

    model = VulFixMinerFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_generator=test_generator)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--finetune_model_path',
                        type=str,
                        default="output/finetuned_model.pt",
                        help='select path to save model')

    args = parser.parse_args()
    
    # ensure path exists
    Path(args.finetune_model_path).parent.mkdir(parents=True, exist_ok=True)

    do_train(args)