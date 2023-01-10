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

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
TEXT = data.Field(tokenize=tokenizer, lower = True)
LABEL = data.LabelField()
train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)
MIN_FREQ = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


TEXT.build_vocab(train_data, 
                min_freq = MIN_FREQ,
                vectors = "glove.6B.300d",
                unk_init = torch.Tensor.normal_,)
LABEL.build_vocab(train_data)
BATCH_SIZE = 512
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
    
class NLIBiLSTM(nn.Module):

    def __init__(self, 
                 input_dim, 
                 embedding_dim,
                 hidden_dim,
                 n_lstm_layers,
                 n_fc_layers,
                 output_dim, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
                                
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.translation = nn.Linear(embedding_dim, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, 
                            hidden_dim, 
                            num_layers = n_lstm_layers, 
                            bidirectional = True, 
                            dropout=dropout if n_lstm_layers > 1 else 0)
        
        fc_dim = hidden_dim * 2
        
        fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(n_fc_layers)]
        
        self.fcs = nn.ModuleList(fcs)
        
        self.fc_out = nn.Linear(fc_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prem, hypo):

        prem_seq_len, batch_size = prem.shape
        hypo_seq_len, _ = hypo.shape
        
        #prem = [prem sent len, batch size]
        #hypo = [hypo sent len, batch size]
        
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)
        
        #embedded_prem = [prem sent len, batch size, embedding dim]
        #embedded_hypo = [hypo sent len, batch size, embedding dim]
        
        translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))
        
        #translated_prem = [prem sent len, batch size, hidden dim]
        #translated_hypo = [hypo sent len, batch size, hidden dim]
        
        outputs_prem, (hidden_prem, cell_prem) = self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(translated_hypo)

        #outputs_x = [sent len, batch size, n directions * hid dim]
        #hidden_x = [n layers * n directions, batch size, hid dim]
        #cell_x = [n layers * n directions, batch size, hid dim]
        
        hidden_prem = torch.cat((hidden_prem[-1], hidden_prem[-2]), dim=-1)
        hidden_hypo = torch.cat((hidden_hypo[-1], hidden_hypo[-2]), dim=-1)
        
        #hidden_x = [batch size, fc dim]

        hidden = torch.cat((hidden_prem, hidden_hypo), dim=1)

        #hidden = [batch size, fc dim * 2]
            
        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden)
        
        #prediction = [batch size, output dim]
        
        return prediction

CUDA_LAUNCH_BLOCKING=1
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
N_LSTM_LAYERS = 2
N_FC_LAYERS = 3
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = NLIBiLSTM(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  N_LSTM_LAYERS,
                  N_FC_LAYERS,
                  OUTPUT_DIM,
                  DROPOUT,
                  PAD_IDX).to(device)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.requires_grad = False

optimizer = optim.AdamW(model.parameters(),)
criterion = nn.CrossEntropyLoss().to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    #a=torch.FloatTensor([y.shape[0]],device=device)    
    #    return correct.sum() / torch.cuda.FloatTensor([y.shape[0]])
    return correct.sum() / torch.FloatTensor([y.shape[0]],device=device)

def evaluate(model, iterator, criterion):
    
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
                
            acc = categorical_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def load(path:str):
    print('Loading model from {}'.format(path))
    model.load_state_dict(torch.load(path+'tut1-model.pt', map_location=device))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')



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

def out(prem:str,hyp:str):
    return predict_inference(prem, hyp, TEXT, LABEL, model, device)