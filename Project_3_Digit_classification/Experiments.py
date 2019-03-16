import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Helpers import prepare_training_data
from Model import ConvNet
import time

input_path = 'C:/Users/User/Desktop/Project_3_Digit_classification'

'''EXPERIMENT'''

# Load and split data into training and validation sets
train_loader, val_loader = prepare_training_data(input_path)

# Set model
model = ConvNet()

# Select loss function
criterion = nn.CrossEntropyLoss()

# Select optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Model training and validation
loss_list_training = []
accuracy_list_training = []
loss_list_validation = []
accuracy_list_validation = []

since = time.time()
for epoch in range(10):

    epoch = epoch + 10
    # TRAINING
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):

        x = Variable(images)
        y_true = Variable(labels)

        # Reset gradients to zero,
        optimizer.zero_grad()

        # Forward propagation
        pred_out = model(x)

        # Compute loss
        loss_training = criterion(pred_out, y_true)

        # Get predicted classes and number of correctly classified instances
        y_pred = torch.max(pred_out.data, 1)[1]
        correct += float((y_pred == y_true).sum())
        total += float(len(y_true))

        # Perform a backward pass, and update the weights
        loss_training.backward()
        optimizer.step()

    # Store loss value on training
    loss_list_training.append(loss_training.data)

    # Calculate accuracy and store on training
    accuracy_training = 100*correct/total
    accuracy_list_training.append(accuracy_training)

    # TESTING on validation set
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(val_loader):

        x = Variable(images)
        y_true = Variable(labels)

        # Forward propagation
        pred_out = model(x)

        # Compute loss
        loss_validation = criterion(pred_out, y_true)

        # Get predicted classes and number of correctly classified instances
        y_pred = torch.max(pred_out.data, 1)[1]
        correct += float((y_pred == y_true).sum())
        total += float(len(y_true))

    # Store loss value on validation
    loss_list_validation.append(loss_validation.data)

    # Calculate accuracy and store on validation
    accuracy_validation = 100*correct/total
    accuracy_list_validation.append(accuracy_validation)

    print(f"Epoch : {epoch}     "
          f"Training Accuracy : {accuracy_training}         "
          f"Validation Accuracy : {accuracy_validation}     "
          f"Training Loss : {loss_training.data.item()}     "
          f"Validation Loss : {loss_validation.data.item()} ")

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
print(model)

# TEMP: loss and accuracy plots
plt.plot(range(len(loss_list_training)), loss_list_training)
plt.plot(range(len(loss_list_validation)), loss_list_validation)
plt.plot(range(len(accuracy_list_training)), accuracy_list_training)
plt.plot(range(len(accuracy_list_validation)), accuracy_list_validation)