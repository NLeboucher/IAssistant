import BiLSTM
import numpy as np
import random
import spacy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from BiLSTM import NLIBiLSTM
import os
import pickle

seed = 1234
BATCH_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
file_path = "acceleration_data"+".pt"
TEXT = data.Field(tokenize=tokenizer, lower=True)
LABEL = data.LabelField()
train_dataset_, valid_dataset_, test_dataset_ =None, None, None
# Load the dataset
if os.path.exists(file_path): #LOADING THE DATA from the pickle file if it were ever created
    train_dataset_, valid_dataset_, test_dataset_ = datasets.SNLI.splits(TEXT, LABEL)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_dataset_, valid_dataset_, test_dataset_), 
    batch_size = BATCH_SIZE,
    device = device)
    with open(file_path, 'rb') as f:
        tempdata_=torch.load(f,)
    (TEXT, LABEL)= tempdata_
else:#LOADING THE DATA from internet download + build (slower)
    train_dataset_, valid_dataset_, test_dataset_ = datasets.SNLI.splits(TEXT, LABEL)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_dataset_, valid_dataset_, test_dataset_), 
        batch_size = BATCH_SIZE,
        device = device)
    MIN_FREQ = 2

    TEXT.build_vocab(train_dataset_, 
                 min_freq=MIN_FREQ,
                 vectors="glove.6B.300d",
                 unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_dataset_)
    with open(file_path, 'wb') as f:
        torch.save((TEXT,LABEL), f)
    






CUDA_LAUNCH_BLOCKING = 1
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
N_LSTM_LAYERS = 2
N_FC_LAYERS = 3
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
N_EPOCHS = 10

model=NLIBiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LSTM_LAYERS, N_FC_LAYERS, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)
optimizer = optim.AdamW(model.parameters(),)
criterion = nn.CrossEntropyLoss().to(device)



# creating iterators for the data 
class TrainNLIBiLSTM():
    def __init__(self, 
                 input_dim=INPUT_DIM, 
                 embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM,
                 n_lstm_layers=N_LSTM_LAYERS,
                 n_fc_layers=N_FC_LAYERS,
                 output_dim=OUTPUT_DIM, 
                 dropout=DROPOUT, 
                 pad_idx=PAD_IDX,
                 batch_size=BATCH_SIZE,
                 device=device,
                 n_epochs=N_EPOCHS):
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.n_fc_layers = n_fc_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.device = device
        self.n_epochs = n_epochs
        
        self.TEXT = TEXT
        self.LABEL = LABEL
        
        self.train_iterator, self.valid_iterator, self.test_iterator = None, None, None
        self.model = None
        
        self.train_losses = []
        self.valid_losses = []
        self.valid_accs = []
        
    def create_iterator(self):
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (train_dataset_, valid_dataset_, test_dataset_), 
            batch_size=self.batch_size,
            device=self.device)

    def create_model(self):
        self.model = model
        
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)
        self.model.embedding.weight.data[self.pad_idx] = torch.zeros(self.embedding_dim)
        self.model.embedding.weight.requires_grad = False
    def _categorical_accuracy(self,preds, y):
        """
        Returns accuracy per batch
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.cuda.FloatTensor([y.shape[0]])
    def _epoch(self, model, iterator, optimizer, criterion): 
        """
        trains model and returns loss and accuracy per epoch
        """
                
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()
        
        for batch in iterator:
            
            prem = batch.premise
            hypo = batch.hypothesis
            labels = batch.label
            
            optimizer.zero_grad()
            
            #prem = [prem sent len, batch size]
            #hypo = [hypo sent len, batch size]
            
            predictions = model(prem, hypo)
            
            #predictions = [batch size, output dim]
            #labels = [batch size]
            
            loss = criterion(predictions, labels)
                    
            acc = self._categorical_accuracy(predictions, labels)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    def _evaluate(self, model, iterator, criterion):
        """
        evaluates model and returns loss and accuracy per epoch
        """        
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                prem = batch.premise
                hypo = batch.hypothesis
                labels = batch.label
                            
                predictions = model(prem, hypo)
                
                loss = criterion(predictions, labels)
                    
                acc = self._categorical_accuracy(predictions, labels)
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    def _save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def _load_model(self, path):
        self.model.load_state_dict(torch.load(path)) 
          
    def train(self):
        best_valid_loss = float('inf')
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_acc = self._epoch(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self._evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self._save_model('tut1-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        

        test_loss, test_acc = self.evaluate(model, test_iterator, criterion)
        return(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    def pretrained(self):
        self._load_model('tut1-model.pt')
        # self.model.load_state_dict(torch.load('tut1-model.pt'))
        test_loss, test_acc = self._evaluate(model, test_iterator, criterion)
        return(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    def predict_inference(premise, hypothesis, text_field, label_field, model, device):
        
        model.eval()
        
        if isinstance(premise, str):
            premise = text_field.tokenize(premise)
        
        if isinstance(hypothesis, str):
            hypothesis = text_field.tokenize(hypothesis)
        
        if text_field.lower:
            premise = [t.lower() for t in premise]
            hypothesis = [t.lower() for t in hypothesis]
            
        premise = [text_field.vocab.stoi[t] for t in premise]
        hypothesis = [text_field.vocab.stoi[t] for t in hypothesis]
        
        premise = torch.LongTensor(premise).unsqueeze(1).to(device)
        hypothesis = torch.LongTensor(hypothesis).unsqueeze(1).to(device)
        
        prediction = model(premise, hypothesis)
        
        prediction = prediction.argmax(dim=-1).item()
        
        return label_field.vocab.itos[prediction]    