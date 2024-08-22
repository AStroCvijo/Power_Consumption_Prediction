import torch
import torch.nn as nn
import torch.optim as optim

# Defining the LSTM model
class LSTMModel(nn.Module):

    # Constructor method
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)

        # Define a fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    # Forward method
    def forward(self, x):
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Pass the last time step output to the fully connected layer
        out = self.fc(lstm_out[:, -1, :])

        return out