from transformers import RobertaTokenizer, RobertaModel
from torch import nn as nn

class VulFixMinerFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VulFixMinerFineTuneClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.1
        self.linear = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]
        x = self.linear(embeddings)
        x = self.drop_out(x)
        out = self.out_proj(x)
   
        return out


class VulFixMinerClassifier(nn.Module):
    def __init__(self):
        super(VulFixMinerClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.1
        self.linear = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, embedding_batch):
        x = self.linear(embedding_batch)
        x = self.drop_out(x)
        out = self.out_proj(x)
   
        return out
