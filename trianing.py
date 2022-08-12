## Default Library import
import os
import json
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import json
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score

## For the purpose of displaying the progress of map function
tqdm.pandas()

## Visualization libraries
import pytesseract
from PIL import Image, ImageDraw

## Specific libraries of LaTr
import torch.nn as nn
from dataset import load_json_file, get_specific_file, resize_align_bbox, get_tokens_with_boxes, create_features
from modeling import LaTr_for_pretraining, LaTr_for_finetuning
from utils import convert_ans_to_token, convert_ques_to_token, rotate, convert_token_to_ques, convert_token_to_answer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import TextVQA
# from query_processing import Question_tokenization, 

## Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

## Warm-up
import pytorch_warmup as warmup
import pytorch_lightning as pl

PAD_TOKEN_BOX = [0, 0, 0, 0]
max_seq_len = 512
batch_size = 1
target_size = (512, 512) ## Note that, ViT would make it 224x224 so :(
t5_model = "t5-base"

train_dataset = pd.read_csv(r"C:\d drive\Director Project\Chargrid VQA\train_v1.0.csv")
train_dataset['answers'] = train_dataset['answers'].apply(lambda x: " ".join(list(map(str, x))))

## Grouping for the purpose of feature extraction
grouped_df = train_dataset.groupby('ucsf_document_id')
## Getting all the unique keys of the group by object
keys = list(grouped_df.groups.keys())

tokenizer = T5Tokenizer.from_pretrained(t5_model)

## Defining the pytorch dataset

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

#Directory to the folders
base_img_path_train = "C:\d drive\Director Project\Chargrid VQA\train\documents"
csv_df_train = "C:\d drive\Director Project\Chargrid VQA\train\train_v1.0.csv"
train_ocr_json_df_train = "C:\d drive\Director Project\Chargrid VQA\train\ocr_csv_folder"


train_ds = TextVQA(base_img_path = base_img_path_train,
                   csv_df = csv_df_train,
                   ocr_json_df = train_ocr_json_df_train,
                   tokenizer = tokenizer,
                   transform = None, 
                   max_seq_length = max_seq_len, 
                   target_size = target_size
                   )

base_img_path_val = "C:\d drive\Director Project\Chargrid VQA\val\documents"
csv_df_val = "C:\d drive\Director Project\Chargrid VQA\val\val_v1.0.csv"
train_ocr_json_df_val = "C:\d drive\Director Project\Chargrid VQA\val\ocr_results"


val_ds = TextVQA(base_img_path = base_img_path_val,
                   csv_df = csv_df_val,
                   ocr_json_df = train_ocr_json_df_val,
                   tokenizer = tokenizer,
                   transform = None, 
                   max_seq_length = max_seq_len, 
                   target_size = target_size
                   )

def collate_fn(data_bunch):

    '''
  A function for the dataloader to return a batch dict of given keys

  data_bunch: List of dictionary
'''

    dict_data_bunch = {}

    for i in data_bunch:
        for (key, value) in i.items():
            if key not in dict_data_bunch:
                dict_data_bunch[key] = []
        dict_data_bunch[key].append(value)

    for key in list(dict_data_bunch.keys()):
        dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis = 0)

    return dict_data_bunch

class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, val_dataset,  batch_size = 1):

        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    
  ## The parameters for Dataloader have been taken from here: https://docs.mosaicml.com/en/v0.7.1/trainer/dataloaders.html#passing-a-pytorch-dataloader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                      collate_fn = collate_fn, shuffle = True, num_workers = 2, pin_memory = True, persistent_workers = True)
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,
                    collate_fn = collate_fn, shuffle = False, num_workers = 2, pin_memory = True, persistent_workers = True)

config = {
    't5_model': 't5-base',
    'vocab_size': 32128,
    'hidden_state': 768,
    'max_2d_position_embeddings': 1001,
    'classes': 32128,  ## number of tokens
    'seq_len': 512
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_acc_score(pred, gt):
    
    ## Function ignores the calculation of padding part
    ## Shape (seq_len, seq_len)
    mask = torch.clamp(gt, min = 0, max = 1)
    last_non_zero_argument = (mask != 0).nonzero()[1][-1]
    pred = pred[:last_non_zero_argument]
    gt = gt[:last_non_zero_argument]  ## Include all the arguments till the first padding index
    
    return accuracy_score(pred, gt)

class LaTrForVQA(pl.LightningModule):
    def __init__(self, config , learning_rate = 1e-4, max_steps = 100000//2):
        super(LaTrForVQA, self).__init__()
        
        self.config = config
        self.save_hyperparameters()
        self.latr =  LaTr_for_finetuning(config)
        self.training_losses = []
        self.validation_losses = []
        self.max_steps = max_steps

#   def configure_optimizers(self):
#     optimizer = torch.optim.AdamW(self.parameters(), lr = self.hparams['learning_rate'])
#     warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = 1000)  
#     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters  = self.max_steps,  verbose = True)
#     return [optimizer], [{"scheduler": (lr_scheduler, warmup_scheduler), "interval": "step"}]

#   def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
#         lr_scheduler, warmup_scheduler = scheduler
#         with warmup_scheduler.dampening():
#                 lr_scheduler.step()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.hparams['learning_rate'])


    def forward(self, batch_dict):
        boxes =   batch_dict['boxes']
        img =     batch_dict['img']
        question = batch_dict['question']
        words =   batch_dict['tokenized_words']
        answer_vector = self.latr(lang_vect = words, 
                                spatial_vect = boxes, 
                                img_vect = img, 
                                quest_vect = question
                                )
        return answer_vector

    def calculate_metrics(self, prediction, labels):

        ## Calculate the accuracy score between the prediction and ground label for a batch, with considering the pad sequence
        batch_size = len(prediction)
        ac_score = 0

        for (pred, gt) in zip(prediction, labels):
            ac_score+= calculate_acc_score(pred.detach().cpu(), gt.detach().cpu())
        ac_score = ac_score/batch_size
        return ac_score

    def training_step(self, batch, batch_idx):
        answer_vector = self.forward(batch)

        ## https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/2
        loss = nn.CrossEntropyLoss()(answer_vector.reshape(-1,self.config['classes']), batch['answer'].reshape(-1))
        _, preds = torch.max(answer_vector, dim = -1)

        ## Calculating the accuracy score
        train_acc = self.calculate_metrics(preds, batch['answer'])
        train_acc = torch.tensor(train_acc)

        ## Logging
        self.log('train_ce_loss', loss,prog_bar = True)
        self.log('train_acc', train_acc, prog_bar = True)
        self.training_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1,self.config['classes']), batch['answer'].reshape(-1))
        _, preds = torch.max(logits, dim = -1)

        ## Validation Accuracy
        val_acc = self.calculate_metrics(preds.cpu(), batch['answer'].cpu())
        val_acc = torch.tensor(val_acc)

        ## Logging
        self.log('val_ce_loss', loss, prog_bar = True)
        self.log('val_acc', val_acc, prog_bar = True)
        
        return {'val_loss': loss, 'val_acc': val_acc}
  ## For the fine-tuning stage, Warm-up period is set to 1,000 steps and again is linearly decayed to zero, pg. 12, of the paper
  ## Refer here: https://github.com/Lightning-AI/lightning/issues/328#issuecomment-550114178
  
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure = None, on_tpu=False,
    using_native_amp=False, using_lbfgs=False):

        ## Warmup for 1000 steps
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        ## Linear Decay
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = polynomial(self.hparams.learning_rate, self.trainer.global_step, max_iter = self.max_steps)

        optimizer.step(opt_closure)
        optimizer.zero_grad()

    def validation_epoch_end(self, outputs):
        
        
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val_loss_epoch_end', val_loss, on_epoch=True, sync_dist=True)
        self.log('val_acc_epoch_end', val_acc, on_epoch=True, sync_dist=True)
        
        self.val_prediction = []
        
#   def training_epoch_end(self, training_step_outputs):
#     train_loss_mean = np.mean(self.training_losses)
#     self.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=self.current_epoch)
#     self.training_losses = []  # reset for next epoch

#   def validation_epoch_end(self, validation_step_outputs):
#     val_loss_mean = np.mean(self.training_losses)
#     self.logger.experiment.add_scalar('validation_loss', val_loss_mean, global_step=self.current_epoch)
#     self.validation_losses = []  # reset for next epoch

url_for_ckpt = 'https://www.kaggleusercontent.com/kf/99663112/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GfuZWkqwWi9nROCTnAS3OQ.YowTb3CNlES2WS_F6BvOSrGs3uLWc2kSBkhElYUcndML0Feuiizdu8trA2e4aj_kdluv1nYlVpS3_86VaJfgSBtyJShQoB0CyxCqdvdMiKl4eQQdWUv2XrTBecEJPXupdFaElzr57CcRjpz35rueyDjf3GVJLznkpSdoyWwSxoxCACbUpS73PKWi97WHfPmEWQgXTDxT_Uno_Pau6fayKyzJ-vWrETzOA2Z6f1-i7umK48D7JBQacS2g_40dW8wIH34QsztCZhHOake7qZnXU_19qaFeDQCNldZ4HcGAmKMtqYI_NK_By370IZ6OHe5Q-mh1f_9SaZoXCzzgaNx4Wsw1THZgzSjZgP2dTLP6a4ZkjHFWiZdkl0azvmoCmSVVYbRdQ9_iI9sFvhUpDWj1bOlr-Zrq9gRi8ksaH9rIzrzk63x_fKPGphZKpxB_l_6iewdGt4yb3GB8kWyGrxBnsGvV5Ei7gTaqv9OAkSKTACMEKB-rj-T8HKtk3ktnEqGMCpHTpkB8RYE6EqYRPbnSYMShjZb12GSn5uYntLtcG7MUbQX-OMt0vzh9fag_zpCyO89K56jxZ6Q9kWdADG0C2T0nR8uC8vWUUBptWNc2tt6pcupcUO19kt7ddNHMbxajHym5AijizrfJbkqnujEodlHWc8C77PawpX2xUPvIlbSvhbdsRRyYfOFGLmZsDdKa.c9dgiKXE5w_-qo4J3He6Qw/models/epoch=0-step=34602.ckpt'

def main():
    datamodule = DataModule(train_ds, val_ds)
    max_steps = 50000       ## 60K Steps
    latr = LaTrForVQA(config, max_steps= max_steps)
    
    try:
        latr = latr.load_from_checkpoint(url_for_ckpt)
        print("Checkpoint loaded correctly")
    except:
        print("Could not load checkpoint")
        return 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_ce_loss", mode="min"
    )
    
    wandb.init(config=config, project="VQA with LaTr")
    wandb_logger = WandbLogger(project="VQA with LaTr", log_model = True, entity="iakarshu")
    
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    pl.seed_everything(42, workers=True)
    
    trainer = pl.Trainer(
        max_steps = max_steps,
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
#         accelerator="tpu",
#         devices=8,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )
    
    trainer.fit(latr, datamodule)