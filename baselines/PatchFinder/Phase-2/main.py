'''
31/08/2023
In this script, we design the model by only using Codereviewer. As you can see, 
the model is based on the pytorch-lightning framework.

04/10/2023
we reuse this script to train the model with the new top100 dataset (colbert).

07/10/2023
we reuse this script to train the model with the new top100 dataset (colbert) with 20 epoch for spliting the dataset.

04/12/2023
we rerun the model to update the metrics.
'''

import configs
from load_data import CVEDataset
# from load_data_colbert import CVEDataset # for baseline usage (ColBERT)
import logging
from torch.utils.data import DataLoader
import os
import wandb



import torch


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set `no_deprecation_warning=True` to disable this warning




"""## Fine-tune using PyTorch Lightning

As we will train the model using PyTorch Lightning, we first need to define a `LightningModule`, which is an `nn.Module` with some additional functionalities. We just need to define the `forward` pass, `training_step` (and optionally `validation_step` and `test_step`), and the corresponding dataloaders.
PyTorch Lightning will then automate the training for us, handling device placement (i.e. we don't need to type `.to(device)` anywhere), etc. It also comes with support for loggers (such as Tensorboard, Weights and Biases) and callbacks.

Of course, you could also train the model in other ways:
* using regular PyTorch
* using the HuggingFace Trainer (in this case, the Seq2SeqTrainer)
* using HuggingFace Accelerate
* etc.
"""

from model import CVEClassifier


"""Let's start up Weights and Biases!"""
if __name__ == '__main__':
    ####### Load the data loaders
    configs.get_singapore_time()
    logging.info('1/4: start to prepare the dataset.')

    train_data = CVEDataset(configs.train_file)
    valid_data = CVEDataset(configs.valid_file)
    test_data = CVEDataset(configs.test_file)



    wandb.login()
    # 8f66cd17219a1912e8a14a65348e656c657f6c5e



    """Next, we initialize the model."""
    configs.get_singapore_time()
    ###### Load the model ######
    logging.info('2/4: start to construct our model.')


    model = CVEClassifier(
            train_dataset=train_data,
            val_dataset=valid_data,
            test_dataset=test_data,
            num_classes=1,   # binary classification
            dropout=0.1
        )


    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    import os


    wandb_logger = WandbLogger(project='PatchFinder')
    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=4,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    CHECK_POINTS_PATH = "./output/all/Checkpoints"

    os.makedirs(CHECK_POINTS_PATH, exist_ok=True)


    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=CHECK_POINTS_PATH,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )


    trainer = Trainer(
        accelerator='gpu',                    # "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto", "ddp"
        num_nodes=1,                          # Using 1 machine/node
        devices=len(configs.gpus),                            # Using 4 GPUs
        default_root_dir=CHECK_POINTS_PATH,   # Default checkpoint path
        logger=wandb_logger,
        max_epochs=20,                        # Set the maximum number of epochs
        accumulate_grad_batches=1,           # Gradient accumulation steps
        max_steps=100000,                     # Set the maximum number of training steps
        log_every_n_steps=10,                # Log every 100 steps
        precision=32,                        # Using 32-bit precision for training; this is the default and can be omitted if desired
        gradient_clip_val=0.0,               # Assuming you don't want gradient clipping, but adjust if needed
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    )
    torch.set_float32_matmul_precision("medium")

    trainer.fit(model)

    """Once we're done training, we can also save the HuggingFace model as follows:"""
    model_save_path = os.path.join(CHECK_POINTS_PATH, "final_model.pt")

    #### So we do not need to load the model Class when evaluate.
    torch.save(model, model_save_path)

    configs.get_singapore_time()


