
# --- Define the CVEClassifier Model (as provided) ---
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy  # Also supports multi-GPU
from torchmetrics.classification import BinaryF1Score  # Optional
from torch.utils.data import DataLoader


class CVEClassifier(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 num_classes=1,
                 dropout=0.1,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder
        
        self.save_hyperparameters()
        self.dropout = dropout
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()
        
        
        # Save datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * self.codeReviewer.config.hidden_size, num_classes)

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg_diff, attention_mask_msg_diff):
        
        # Get [CLS] embeddings for desc and msg+diff
        desc_cls_embed = self.codeReviewer(input_ids=input_ids_desc, attention_mask=attention_mask_desc).last_hidden_state[:, 0, :]
        msg_diff_cls_embed = self.codeReviewer(input_ids=input_ids_msg_diff, attention_mask=attention_mask_msg_diff).last_hidden_state[:, 0, :]
        
        # Concatenate [CLS] embeddings
        concatenated = torch.cat((desc_cls_embed, msg_diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output
    
    def common_step(self, batch):
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg_diff'],
            batch['attention_mask_msg_diff']
        ).squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss, predict, batch['label']

    def training_step(self, batch, dataloader_idx=None):
        loss, logits, labels = self.common_step(batch)
        preds = torch.sigmoid(logits) > 0.5
        self.train_accuracy.update(preds, labels.int())
        self.train_f1.update(preds, labels.int())
        
        self.log("training_loss", loss, batch_size=batch['input_ids_desc'].shape[0])
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits, labels = self.common_step(batch)
        preds = torch.sigmoid(logits) > 0.5
        self.val_accuracy.update(preds, labels.int())
        self.val_f1.update(preds, labels.int())
        
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch['input_ids_desc'].shape[0])
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits, labels = self.common_step(batch)
        preds = torch.sigmoid(logits) > 0.5
        self.test_accuracy.update(preds, labels.int())
        self.test_f1.update(preds, labels.int())
    
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
    
        return loss
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        total_steps = self.hparams.num_train_epochs * len(self.train_dataloader())
        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=total_steps
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=32, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, num_workers=15)