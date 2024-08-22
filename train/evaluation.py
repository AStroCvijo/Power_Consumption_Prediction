import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Function for evaluating the model
def model_evaluate(model, test_loader, criterion, device):

    # Switch to evaluation mode
    model.eval()

    # Disable the gradient
    with torch.no_grad():

        # Initialize test loss
        test_loss = 0.0

        # Initialize empty predictions and targets arrays
        all_predictions = []
        all_targets = []

        for inputs, targets in test_loader:
            
            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets.unsqueeze(1))

            # Accumulate the loss
            test_loss += loss.item()

            # Store predictions and targets for further analysis
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Calculate average loss over the test set
        test_loss /= len(test_loader)

        # Convert predictions and targets to numpy arrays for further calculations
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(all_predictions - all_targets))

    # Print Test Loss and MAE
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")