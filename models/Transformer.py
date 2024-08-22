import torch
import torch.nn as nn
import torch.optim as optim

# Defining the Transformer model
class TransformerModel(nn.Module):

    # Constructor method
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()

        # Define the input projection layer
        self.input_fc = nn.Linear(input_size, d_model)

        # Define the Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )

        # Define the fully connected output layer
        self.fc = nn.Linear(d_model, output_size)

    # Forward method
    def forward(self, x):
        
        # Project the input to the model dimension (d_model)
        x = self.input_fc(x)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)

        # Pass the last time step output to the fully connected layer
        out = self.fc(x[:, -1, :])

        return out
