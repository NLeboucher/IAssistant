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
        
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)

        translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))

        outputs_prem, (hidden_prem, cell_prem) = self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(translated_hypo)

        hidden_prem = torch.cat((hidden_prem[-1], hidden_prem[-2]), dim=-1)
        hidden_hypo = torch.cat((hidden_hypo[-1], hidden_hypo[-2]), dim=-1)
        
        hidden = torch.cat((hidden_prem, hidden_hypo), dim=1)

        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden)
        
        return prediction
    
class MyBiLSTM(NLIBiLSTM):
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensures that the same sequence of random numbers will be generated every time.
# Sets the seed for generating random numbers in a torch.Generator object.

    def __init__(self,
                    INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    N_LSTM_LAYERS,
                    N_FC_LAYERS,
                    OUTPUT_DIM,
                    PAD_IDX,
                    DROPOUT=0.25,
                    N_EPOCHS=10,
                    device=device,
                 ) -> None:
        super().__init__(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  N_LSTM_LAYERS,
                  N_FC_LAYERS,
                  OUTPUT_DIM,
                  DROPOUT,
                  PAD_IDX)
        
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        TEXT = data.Field(tokenize=tokenizer, lower = True)
        LABEL = data.LabelField()
        train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)
        MIN_FREQ = 2

        TEXT.build_vocab(train_data, 
                        min_freq = MIN_FREQ,
                        vectors = "glove.6B.300d",
                        unk_init = torch.Tensor.normal_)

        LABEL.build_vocab(train_data)
        BATCH_SIZE = 64
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        model = NLIBiLSTM(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  N_LSTM_LAYERS,
                  N_FC_LAYERS,
                  OUTPUT_DIM,
                  DROPOUT,
                  PAD_IDX).to(device)
        optimizer = optim.AdamW(model.parameters(),)
        criterion = nn.CrossEntropyLoss().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def categorical_accuracy(preds, y):
        """
        Returns accuracy per batch
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.cuda.FloatTensor([y.shape[0]])
    def _train(model, iterator, optimizer, criterion):

        
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
                    
            acc = categorical_accuracy(predictions, labels)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
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
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    def monitoredtraining(best_valid_loss = float('inf')):
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_acc = train(model, self.train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut1-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    def test(model, test_iterator, criterion):
        model.load_state_dict(torch.load('tut1-model.pt'))
        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    def _predict_inference(premise, hypothesis, text_field, label_field, model, device):
    
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
a=MyBiLSTM()