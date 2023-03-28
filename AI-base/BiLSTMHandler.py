from BiLSTMTraining import TrainNLIBiLSTM
import pandas as pd
from torchtext import datasets
import torch.optim as optim
import torch.nn as nn


BATCH_SIZE = 512
INPUT_DIM = 23551
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
N_LSTM_LAYERS = 2
N_FC_LAYERS = 3
OUTPUT_DIM = 3
DROPOUT = 0.25
PAD_IDX = 1
N_EPOCHS = 10
# Initialize the NLIBiLSTM model
trainer=TrainNLIBiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LSTM_LAYERS, N_FC_LAYERS, OUTPUT_DIM, DROPOUT, PAD_IDX, BATCH_SIZE, N_EPOCHS)
# Create the iterator
trainer.create_iterator()
# Create the model
trainer.create_model()
#training
trainer.train()
# Save the model
trainer._save_model("NLIBiLSTM.pt")

