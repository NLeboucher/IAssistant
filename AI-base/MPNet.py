from datasets import load_dataset, load_metric
import datasets
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import random
import torch
from torch.optim import Adam

import transformers
from transformers import AdamW,AutoTokenizer, AutoModelForSequenceClassification,get_linear_schedule_with_warmup, TrainingArguments, Trainer
from torch.optim import Adadelta, Adam, SGD, RMSprop, AdamW
import wandb
#from transformers.trainer_utils import A
#%pip install scikit-learn
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("using device:", device)

class MyMPNet(): #torch.nn.Module
    def __init__(self,
            #Training Arguments
        model_checkpoint = "microsoft/mpnet-base",#"symanto/mpnet-base-snli-mnli" got 87% on MNLI
        model = AutoModelForSequenceClassification.from_pretrained, # for mnli NO MORE THIS WRITING METHOD TRY TO USE THIS AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
        #quite a reorganiser for the code
        METRIC_NAME = "accuracy",
        BATCHSIZE = 4,
        EPOCHNUM = 0.20,
        WEIGHT_DECAY = 0.8,

        LEARNING_RATE = 6.7e-5,

        tokenizer=AutoTokenizer.from_pretrained,
        dataset = load_dataset("glue", "mnli"),
        metric = load_metric('glue', "mnli"),
        num_labels = 3, # for mnli
        optimizer_name = "AdamW",
        sentence1_key = "premise", # for mnli
        sentence2_key ="hypothesis", # for mnli
    ):
        #super(MyMPNet, self).__init__()
        self.model = model(model_checkpoint, num_labels=num_labels)
        self.tokenizer = tokenizer(model_checkpoint, use_fast=True) # for mnli
        optimizer_name = optimizer_name
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.metric = metric
        self.dataset = dataset
        MODEL_NAME = model_checkpoint.split("/")[-1],

        self.args = TrainingArguments(
                f"{MODEL_NAME}-finetuned-{optimizer_name}-mnli",
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=LEARNING_RATE,#6.7e-5(81.7),#1.8e-5,#2.4e-5 , #3e-5 AdamW 
                per_device_train_batch_size=BATCHSIZE,
                per_device_eval_batch_size=BATCHSIZE,
                num_train_epochs=EPOCHNUM,                                          
                weight_decay=WEIGHT_DECAY,                                               
                load_best_model_at_end=True,
                metric_for_best_model=METRIC_NAME,
                push_to_hub=True,
                report_to="wandb",
            )
        encoded_dataset = dataset.map(MyMPNet.preprocess_function, batched=True)
        
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation_matched"],#for mnli
            tokenizer=self.tokenizer,
            compute_metrics=MyMPNet.compute_metrics,
        )

    def preprocess_function(self,examples):
        return self.tokenizer(
            examples[self.sentence1_key],
            examples[self.sentence2_key],
            padding=True,
            truncation=True
            )
    def compute_metrics(self,eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)
    def train(self):
        self.trainer.train()

