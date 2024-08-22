import torch
import torch.nn as nn
import torch.optim as optim

# Defining the GRU model
class GRUModel(nn.Module):

    # Constructor method
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        # Define the GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Define a fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    # Forward method
    def forward(self, x):

        # GRU layer
        gru_out, _ = self.gru(x)

        # Pass the last time step output to the fully connected layer
        out = self.fc(gru_out[:, -1, :])

        return out
