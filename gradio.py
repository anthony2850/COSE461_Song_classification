from datasets import load_dataset
import ast

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import json
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, tqdm_notebook

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import gradio as gr

training_config = {
    "model_name" : "beomi/kcbert-large",
    # "mode" : "train",
    # "mode" : "finetuning",
    "mode" : "test",
    # "checkpoint_path" : None,
    # "checkpoint_path" : "/home/zheedong/Projects/NLP/lightning_logs/kcbert_large_100epoch/checkpoints/last.ckpt",
    "checkpoint_path" : "/home/zheedong/Projects/NLP/lightning_logs/kcbert_large_init_bert_100epoch_v3/checkpoints/model-epoch=21-val_loss=1.33.ckpt",
    "max_len" : 256,
    # "batch_size" : 128,
    "batch_size" : 1,
    "warmup_ratio" : 0.1,
    "num_epochs" : 50,
    "max_grad_norm" : 1,
    "log_interval" : 1,
    "learning_rate" : 1e-5,
    # "learning_rate" : 5e-6,
    # "gradient_accumulation" : 4,
    "gradient_accumulation" : 1,
    # "devices" : 4,
    "devices" : 1,
}

class BERTClassifier(pl.LightningModule):
    def __init__(self,
                 bert,
                 tokenizer,
                 config,
        ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        try:
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        except:
            return self.bert(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )

    def classify_text(self, text):
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Feed the inputs into the model
        outputs = self.bert(input_ids, attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        # Map the predicted label to the corresponding emotion
        emotions = {
            0: "분노",
            1: "슬픔",
            2: "불안",
            3: "상처",
            4: "당황",
            5: "기쁨"
        }

        predicted_emotion = emotions[predicted_label]

        return predicted_emotion

    def training_step(self, batch, batch_idx):
        text, label = batch
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']

        out = self(input_ids, attention_mask)
        loss = self.loss_fn(out['logits'], label)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        input_ids = text['input_ids'].squeeze()
        attention_mask = text['attention_mask'].squeeze()
        out = self(input_ids, attention_mask)
        loss = self.loss_fn(out['logits'], label)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        preds = torch.argmax(out['logits'], dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.trainer.is_global_zero:
            print(f"{'='*10}val_acc: {acc}{10*'='}")

    def test_step(self, batch, batch_idx):
        text, label = batch
        input_ids = text['input_ids'].squeeze()
        attention_mask = text['attention_mask'].squeeze()
        out = self(input_ids, attention_mask)
        preds = torch.argmax(out['logits'], dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log('test_acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.trainer.is_global_zero:
            print(f"{'='*10}test_acc: {acc}{10*'='}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=5e-2
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_ratio'] * self.config['num_epochs'],
            num_training_steps=self.config['num_epochs'],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


if __name__ == '__main__':
    # train_data_path = '/content/drive/MyDrive/대학 수업/COSE461 자연어처리/감성대화/Training/라벨링데이터/감성대화말뭉치(최종데이터)_Training.json'
    # train_data_path = 'data/감성대화/Training/라벨링데이터/감성대화말뭉치(최종데이터)_Training.json'
    train_data_path = '/home/zheedong/Projects/NLP/song_train.csv'
    # val_data_path = '/content/drive/MyDrive/대학 수업/COSE461 자연어처리/감성대화/Validation/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json'
    # val_data_path = 'data/감성대화/Validation/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json'
    val_data_path = '/home/zheedong/Projects/NLP/song_validation.csv'

    model_name = training_config['model_name']
    # bertmodel, vocab = get_pytorch_kobert_model()
    bertmodel = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = BERTClassifier.load_from_checkpoint(
        bert=bertmodel,
        tokenizer=tokenizer,
        config=training_config,
        checkpoint_path=training_config['checkpoint_path']
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    gr.Interface(
        model.classify_text,
        inputs=gr.inputs.Textbox(lines=5, label="노래 가사를 입력해 주세요"),
        outputs="text",
        title="노래 가사 감정 분석",
        description="노래 가사를 입력하고 제가 생각하는 감정을 확인해 보세요! 🎶🎵"
    ).launch(share=True)
    