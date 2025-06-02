from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel
import os
import json
import torch

hunk_data_folder_name = 'hunk_data'
file_data_folder_name = 'variant_file_data'

directory = os.path.dirname(os.path.abspath(__file__))
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
empty_code = tokenizer.sep_token + ''
inputs = tokenizer([empty_code], padding=True, max_length=512, truncation=True, return_tensors="pt")
input_ids, attention_mask = inputs.data['input_ids'], inputs.data['attention_mask']
empty_embedding = code_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0, 0, :].tolist()

class VulFixMinerDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_embedding, id_to_url):
        self.list_IDs = list_IDs
        self.labels = labels
        self.id_to_url = id_to_url
        self.id_to_embedding = id_to_embedding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        url = self.id_to_url[id]
        embedding = self.id_to_embedding[id]
        y = self.labels[id]

        return int(id), url, embedding, y  


class VulFixMinerFileDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_url, id_to_input, id_to_mask):
        self.list_IDs = list_IDs
        self.labels = labels
        self.id_to_url = id_to_url
        self.id_to_input = id_to_input
        self.id_to_mask = id_to_mask

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        url = self.id_to_url[id]
        input_id = self.id_to_input[id]
        mask = self.id_to_mask[id]

        y = self.labels[id]

        return int(id), url, input_id, mask, y    

