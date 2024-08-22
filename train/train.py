import torch
import torch.nn as nn
import torch.optim as optim

# Function for training the model
def model_train(model, criterion, optimizer, num_epochs, train_loader, device, model_name):

    for epoch in range(num_epochs):

        # Set the model to training and initialize current loss
        model.train()
        curr_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):

            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero out the gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            curr_loss += loss.item()

        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {curr_loss/len(train_loader):.4f}")

    print("Finished training")
    
    # Save the model
    model_path = 'pretrained_models/' + model_name + '.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}\n')