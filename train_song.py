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
from torchmetrics import Precision, Recall, F1Score

training_config = {
    "model_name" : "beomi/kcbert-large",
    # "mode" : "train",
    # "mode" : "finetuning",
    "mode" : "test",
    # "checkpoint_path" : None,
    # "checkpoint_path" : "/home/zheedong/Projects/NLP/lightning_logs/kcbert_large_100epoch/checkpoints/last.ckpt",
    # "checkpoint_path" : "/home/zheedong/Projects/NLP/lightning_logs/kcbert_large_init_bert_100epoch_v3/checkpoints/model-epoch=21-val_loss=1.33.ckpt",
    "checkpoint_path" : "/home/zheedong/Projects/NLP/lightning_logs/kcbert_large_init_bert_pretrain_v3/checkpoints/model-epoch=18-val_loss=1.85.ckpt",
    "max_len" : 256,
    "batch_size" : 128,
    # "batch_size" : 1,
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

class SongDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len):
        self.dataset = dataset
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

    def get_flatten_text(self, train_content):
        return " ".join(ast.literal_eval(train_content))

    def convert_type_to_class(self, data_type):
        if data_type.startswith('E1') or data_type == "분노":
            return 0
        elif data_type.startswith('E2') or data_type == "슬픔":
            return 1
        elif data_type.startswith('E3') or data_type == "불안":
            return 2
        elif data_type.startswith('E4') or data_type == "상처":
            return 3
        elif data_type.startswith('E5') or data_type == "당황":
            return 4
        elif data_type.startswith('E6') or data_type == "기쁨":
            return 5
        else:
            raise ValueError("Invalid emotion type")

    def __getitem__(self, idx):
        try:
            text = "[CLS]" + self.get_flatten_text(self.dataset['lyric'][idx])
        except:
            dirty_text = self.dataset['lyric'][idx]
            clean_text = ''.join(ch for ch in dirty_text if '\uAC00' <= ch <= '\uD7A3' or ch.isalnum() or ch.isspace())
            # if invaild character in lyric, replace it with space
            text = "[CLS]" + clean_text
        label = self.convert_type_to_class(self.dataset['emotion'][idx])
        text_token = self.bert_tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt',
            padding='max_length',
        )
        text_token['input_ids'] = text_token['input_ids'].squeeze()
        text_token['attention_mask'] = text_token['attention_mask'].squeeze()
        return text_token, label

    def __len__(self):
        return len(self.dataset)

class BERTDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, tokenizer, trainig_config):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.tokenizer = tokenizer
        self.max_len = training_config['max_len']
        self.batch_size = training_config['batch_size']

        self.train_data = load_dataset("csv", data_files=train_data_path)['train']
        self.val_data = load_dataset("csv", data_files=val_data_path)['train']

    def setup(self, stage=None):
        self.train_dataset = SongDataset(self.train_data, self.tokenizer, self.max_len)
        self.val_dataset = SongDataset(self.val_data, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

class BERTClassifier(pl.LightningModule):
    def __init__(self,
                 bert,
                 config,
        ):
        super().__init__()
        self.bert = bert
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

        # For metrics
        self.precision = Precision(num_classes=6, average='macro', task='MULTICLASS')
        self.recall = Recall(num_classes=6, average='macro', task='MULTICLASS')
        self.f1 = F1Score(num_classes=6, average='macro', task='MULTICLASS')

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
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
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

        self.precision.update(preds, label)
        self.recall.update(preds, label)
        self.f1.update(preds, label)

    def on_test_epoch_end(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        if self.trainer.is_global_zero:
            print(f"{'='*10}test_precision: {precision}{10*'='}")
            print(f"{'='*10}test_recall: {recall}{10*'='}")
            print(f"{'='*10}test_f1: {f1}{10*'='}")

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
    vocab = tokenizer.get_vocab()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    datamodule = BERTDataModule(
        train_data_path,
        val_data_path,
        tokenizer,
        training_config
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.val_dataloader()

    model_summary = pl.callbacks.ModelSummary(max_depth=1)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=5, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        filename='model-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=training_config['devices'],
        strategy="ddp",
        max_epochs=training_config['num_epochs'],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=4,
        callbacks=[model_summary, lr_logger, checkpoint_callback], #early_stopping],
        gradient_clip_val=5,
        accumulate_grad_batches=training_config['gradient_accumulation'],
        precision='bf16',
        enable_checkpointing=True,
        deterministic=True,
    )

    if training_config["mode"] == "train":
        print(f"Using original model")
        model = BERTClassifier(bertmodel, config=training_config)
        trainer.fit(model, train_dataloader, test_dataloader)
    elif training_config["mode"] == "finetuning":
        print(f"Using checkpoint model")
        model = BERTClassifier.load_from_checkpoint(
            bert=bertmodel,
            config=training_config,
            checkpoint_path=training_config["checkpoint_path"]
        )
        trainer.fit(model, train_dataloader, test_dataloader)
    elif training_config["mode"] == "test":
        if training_config["checkpoint_path"] is None:
            print(f"Using original model")
            model = BERTClassifier(bertmodel, config=training_config)
        else:
            print(f"Using checkpoint model")
            model = BERTClassifier.load_from_checkpoint(
                bert=bertmodel,
                config=training_config,
                checkpoint_path=training_config["checkpoint_path"]
            )
        trainer.test(model, test_dataloader)
    else:
        raise ValueError("Invalid mode")