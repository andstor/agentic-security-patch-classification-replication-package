import torch
from torch import nn as nn
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from torch.optim import AdamW
from transformers import get_scheduler
from patch_entities import VulFixMinerDataset
from model import VulFixMinerClassifier, VulFixMinerFineTuneClassifier
from tqdm import tqdm
import argparse
import vulfixminer_finetune
from transformers import RobertaTokenizer, RobertaModel
import csv 
from utils import extract_file_diffs, get_code_version

# dataset_name = 'sap_patch_dataset.csv'
# EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2'
# MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'

dataset_name = None
FINETUNE_MODEL_PATH = None
MODEL_PATH = None
TRAIN_PROB_PATH = None
TEST_PROB_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(directory, 'model')


# retest with SAP dataset
NUMBER_OF_EPOCHS = 20
EARLY_STOPPING_ROUND = 5

TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
use_mps = torch.backends.mps.is_available()
device = torch.device("cuda:0" if use_cuda else "mps" if use_mps else "cpu")

torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 256
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2


# model_path_prefix = model_folder_path + '/patch_variant_2_16112021_model_'


def predict_test_data(model, testing_generator, device, need_prob=False, prob_path=None):
    y_pred = []
    y_test = []
    probs = []
    losses = []
    loss_function = nn.NLLLoss()
    urls = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(testing_generator, dynamic_ncols=True, leave=True):
            
            embedding_batch = batch["embeddings"].to(device)
            label_batch = batch["labels"].to(device)

            outs = model(embedding_batch)
            outs_softmax = F.softmax(outs, dim=1)
            outs_logsoftmax = F.log_softmax(outs, dim=1)
            
            loss = loss_function(outs_logsoftmax, label_batch)
            losses.append(loss.item())
            y_pred.extend(torch.argmax(outs_softmax, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs_softmax[:, 1].tolist())
            
            urls.extend(batch["urls"])

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        avg_loss = np.mean(losses) if losses else 0
        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")

    if prob_path is not None:
        with open(prob_path, 'w') as file:
            writer = csv.writer(file)
            for i, prob in enumerate(probs):
                writer.writerow([urls[i], prob])

    if not need_prob:
        return precision, recall, f1, auc, avg_loss
    else:
        return precision, recall, f1, auc, urls, probs, avg_loss


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
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
        progress_bar.set_description(f"Epoch [{epoch+1}/{number_of_epochs}]")
        
        for batch in training_generator:
            
            
            embedding_batch = batch["embeddings"].to(device)
            label_batch = batch["labels"].to(device)

            outs = model(embedding_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            
            avg_loss = np.mean(train_losses)
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": total_loss,
                "avg_loss": avg_loss
            })

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        model.eval()

        print("Result on testing dataset...")
        precision, recall, f1, auc, val_loss = predict_test_data(model=model,
                                                       testing_generator=test_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        if val_loss < best_val_loss: # 
                best_val_loss = val_loss
                epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_ROUND:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    progress_bar.close()
    
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), MODEL_PATH)
    else:
        torch.save(model.state_dict(), MODEL_PATH)

    return model


class CommitAggregator:
    def __init__(self, file_transformer):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.file_transformer = file_transformer

    def transform(self, diff_list):
        # cap at 20 file diffs 
        diff_list = diff_list[:20]
        input_list, mask_list = [], []
        for diff in diff_list:
            added_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=True)
            deleted_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=False)

            code = added_code + self.tokenizer.sep_token + deleted_code
            input_ids, mask = vulfixminer_finetune.get_input_and_mask(self.tokenizer, [code])
            input_list.append(input_ids)
            mask_list.append(mask)

        input_list = torch.stack(input_list)
        mask_list = torch.stack(mask_list)
        input_list, mask_list = input_list.to(device), mask_list.to(device)
        embeddings = self.file_transformer(input_list, mask_list).last_hidden_state[:, 0, :]

        sum_ = torch.sum(embeddings, dim=0)
        mean_ = torch.div(sum_, len(diff_list))
        mean_ = mean_.detach()
        mean_ = mean_.cpu()

        return mean_

def collate_fn(batch):
    labels = torch.stack([item['label'] for item in batch])
    embeddings = torch.stack([item['embedding'] for item in batch])
    urls = [item['url'] for item in batch]
    return {'labels': labels, 'embeddings': embeddings, 'urls': urls}


def do_train(args):
    global dataset_name, MODEL_PATH

    dataset_name = args.dataset_path
    FINETUNE_MODEL_PATH = args.finetune_model_path
    MODEL_PATH = args.model_path

    TRAIN_PROB_PATH = args.train_prob_path
    TEST_PROB_PATH = args.test_prob_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(MODEL_PATH))

    print("Loading finetuned file transformer...")
    finetune_model = VulFixMinerFineTuneClassifier()
    
    finetune_model.load_state_dict(torch.load(FINETUNE_MODEL_PATH))

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        finetune_model = nn.DataParallel(finetune_model)
    
    if hasattr(finetune_model, "module"):
        code_bert = finetune_model.module.code_bert
    else:
        code_bert = finetune_model.code_bert
        
    code_bert.eval()
    code_bert.to(device)

    print("Finished loading model")


    ds_commits = vulfixminer_finetune.load_data()
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    def construct_urls(row):
        url = row["owner"] + '/' + row["repo"] + '/commit/' + row["commit_id"]
        row["url"] = url
        return row
    
    ds_commits = ds_commits.map(
        construct_urls,
        batched=False,
        batch_size=1,
        num_proc=10,
        desc="Constructing URLs for commits"
    )
    
    
    ds_diffs = ds_commits.map(
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
        res = tokenizer(examples["code"], add_special_tokens=True, max_length=CODE_LENGTH, truncation=True, padding="max_length")
        res["label"] = examples["label"]
        return res

    ds_tokenized = ds_code.map(preprocess_function, batched=False, num_proc=1, desc="Tokenizing")#, remove_columns=ds_code["train"].column_names)
    ds_tokenized = ds_tokenized.with_format("torch")

    def get_embeddings(row):
        with torch.no_grad():
            inputs = row['input_ids'].to(device)
            attention = row['attention_mask'].to(device)
            outputs = code_bert(inputs, attention)
            embeddings = outputs.last_hidden_state[:, 0, :]
            #sum_ = torch.sum(embeddings, dim=0)
            #mean_ = torch.div(sum_, len(row['input_ids']))
            mean_ = torch.mean(embeddings, dim=0).cpu()
        
        row["embedding"] = mean_

        return row


    predictions = ds_tokenized.map(
        get_embeddings,
        batched=False,
        num_proc=1,
        desc="Computing embeddings",
    )
    #ds_predictions = predictions.remove_columns(["input_ids", "attention_mask", "code"])
    #ds_predictions = ds_predictions.rename_columns({"embeddings": "input_ids"})
    
    ds = predictions.with_format("torch", columns=["embedding", "label"], output_all_columns=True)
    
    
    training_generator = DataLoader(
        ds["train"],
        collate_fn=collate_fn,
        **TRAIN_PARAMS
    )
    
    test_generator = DataLoader(
        ds["test"],
        collate_fn=collate_fn,
        **TEST_PARAMS
    )
   
    model = VulFixMinerClassifier()

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

    print("Writing result to file...")
    predict_test_data(model=model, testing_generator=training_generator, device=device, prob_path=TRAIN_PROB_PATH)
    predict_test_data(model=model, testing_generator=test_generator, device=device, prob_path=TEST_PROB_PATH)
    print("Finish writting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        default="fals3/cvevc_commits",
                        help='name of dataset')
    parser.add_argument('--model_path',
                        type=str,
                        default="output/trained_model.pt",
                        help='save train model to path')

    parser.add_argument('--finetune_model_path',
                        type=str,
                        default="output/finetuned_model.pt",
                        help='path to finetune file transfomer')

    parser.add_argument('--train_prob_path',
                        type=str,
                        default="output/train_prob.csv",
                        help='')

    parser.add_argument('--test_prob_path',
                        type=str,
                        default="output/test_prob.csv",
                        help='')
   
    args = parser.parse_args()


    do_train(args)