import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Import models
from models.LSTM import LSTMModel
from models.GRU import GRUModel
from models.Transformer import TransformerModel

# Import functions for data preprocessing
from data.data_functions import data_preprocess
from data.data_functions import create_sequences
from data.data_functions import create_data_loaders

# Import functions for model training and evaluation
from train.train import model_train
from train.evaluation import model_evaluate

# Import argument parser
from utils.argparser import arg_parse

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    data = data_preprocess("data/powerconsumption.csv")

    # Extract sequences and targets from data
    sequences, targets = create_sequences(data, args.sequence_length, args.prediction_step, args.prediction_target)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(sequences, targets)

    # Train the model with given parameters
    if args.train:

        # Initialize the model
        if args.model == 'LSTM':

            # Parameters for the model
            input_size = sequences.shape[2]    # Number of features
            hidden_size = args.hidden_size     # Number of features in the hidden state
            num_layers = args.number_of_layers # Number of recurrent layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Training the LSTM model")
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        elif args.model == 'GRU':

            # Parameters for the model
            input_size = sequences.shape[2]    # Number of features
            hidden_size = args.hidden_size     # Number of features in the hidden state
            num_layers = args.number_of_layers # Number of recurrent layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Training the GRU model")
            model = GRUModel(input_size, hidden_size, num_layers, output_size)

        elif args.model == 'Transformer':

            # Parameters for the Transformer model
            input_size = sequences.shape[2]    # Number of features
            d_model = args.model_dimension     # Transformer model dimension
            nhead = args.attention_heads       # Number of attention heads
            num_layers = args.number_of_layers # Number of transformer encoder layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Training the Transformer model")
            model = TransformerModel(input_size, d_model, nhead, num_layers, output_size)

        # Move the model to the device
        model.to(device)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        num_epochs = args.epochs

        # Get the name of the model
        model_name = args.model
        if args.model_name != '':
            model_name = args.model_name

        # Train the model
        model_train(model, criterion, optimizer, num_epochs, train_loader, device, model_name)

        # Evaluate the model
        model_evaluate(model, test_loader, criterion, device)

    # Load the model from the chosen directory
    elif args.load != '':

        # Initialize the model
        if args.model == 'LSTM':

            # Parameters for the model
            input_size = sequences.shape[2]    # Number of features
            hidden_size = args.hidden_size     # Number of features in the hidden state
            num_layers = args.number_of_layers # Number of recurrent layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Using the LSTM model")
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        elif args.model == 'GRU':

            # Parameters for the model
            input_size = sequences.shape[2]    # Number of features
            hidden_size = args.hidden_size     # Number of features in the hidden state
            num_layers = args.number_of_layers # Number of recurrent layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Using the GRU model")
            model = GRUModel(input_size, hidden_size, num_layers, output_size)

        elif args.model == 'Transformer':

            # Parameters for the Transformer model
            input_size = sequences.shape[2]    # Number of features
            d_model = args.model_dimension     # Transformer model dimension
            nhead = args.attention_heads       # Number of attention heads
            num_layers = args.number_of_layers # Number of transformer encoder layers
            output_size = 1                    # Number of outputs (Power Consumption)

            print("Using the Transformer model")
            model = TransformerModel(input_size, d_model, nhead, num_layers, output_size)

        # Load the model
        model.load_state_dict(torch.load(args.load))
        print(f'Model loaded from {args.load}\n')

        # Move the model to the device
        model.to(device)

        # Evaluate the model
        model.eval()
        criterion = nn.MSELoss()
        model_evaluate(model, test_loader, criterion, device)
