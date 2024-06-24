from datasets import load_dataset
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

training_config = {
    "max_len" : 128,
    "batch_size" : 256,
    "warmup_ratio" : 0.1,
    "num_epochs" : 100,
    "max_grad_norm" : 1,
    "log_interval" : 1,
    "learning_rate" : 1e-5,
    "gradient_accumulation" : 4,
}

class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len):
        self.dataset = dataset
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

    def get_flatten_text(self, train_content):
        return "\n".join(list(train_content.values()))

    def convert_type_to_class(self, data_type):
        if data_type.startswith('E1'):
            return 0
        elif data_type.startswith('E2'):
            return 1
        elif data_type.startswith('E3'):
            return 2
        elif data_type.startswith('E4'):
            return 3
        elif data_type.startswith('E5'):
            return 4
        elif data_type.startswith('E6'):
            return 5
        else:
            raise ValueError("Invalid emotion type")

    def __getitem__(self, idx):
        text = "[CLS]" + self.get_flatten_text(self.dataset[idx]['talk']['content'])
        label = self.convert_type_to_class(self.dataset[idx]['profile']['emotion']['type'])
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
        with open(train_data_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        with open(val_data_path, 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)

    def setup(self, stage=None):
        self.train_dataset = BERTDataset(self.train_data, self.tokenizer, self.max_len)
        self.val_dataset = BERTDataset(self.val_data, self.tokenizer, self.max_len)

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

    def forward(self, input_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
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
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        preds = torch.argmax(out['logits'], dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['learning_rate'])
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
    train_data_path = 'data/감성대화/Training/라벨링데이터/감성대화말뭉치(최종데이터)_Training.json'
    # val_data_path = '/content/drive/MyDrive/대학 수업/COSE461 자연어처리/감성대화/Validation/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json'
    val_data_path = 'data/감성대화/Validation/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json'
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    "\n".join(list(train_data[1]['talk']['content'].values()))

    model_name = "beomi/kcbert-large"
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
        monitor='val_loss',
        mode='min',
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        filename='model-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy="ddp",
        max_epochs=training_config['num_epochs'],
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=1,
        callbacks=[model_summary, lr_logger, early_stopping, checkpoint_callback],
        gradient_clip_val=5,
        accumulate_grad_batches=training_config['gradient_accumulation'],
        precision='bf16',
    )

    model = BERTClassifier(bertmodel, config=training_config)
    trainer.fit(model, train_dataloader, test_dataloader)