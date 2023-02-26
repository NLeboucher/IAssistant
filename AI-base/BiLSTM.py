import torch.nn as nn
import torch
import torch.nn.functional as F

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