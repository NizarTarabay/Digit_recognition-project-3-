import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

from Model import ConvNet
from Helpers import prepare_submission_file, prepare_submission_data
'''SUBMISSION'''

# Load whole training data and test set
train_loader, test_images = prepare_submission_data(dir)

# Set model
model = ConvNet()

# Select loss function
criterion = nn.CrossEntropyLoss()

# Select optimizer
optimizer = torch.optim.Adam(model.parameters())

loss_list = []
accuracy_list = []
since = time.time()
for epoch in range(10):

    # Training
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
        loss = criterion(pred_out, y_true)

        # Get predicted classes and number of correctly classified instances
        y_pred = torch.max(pred_out.data, 1)[1]
        correct += float((y_pred == y_true).sum())
        total += float(len(y_true))

        # Perform a backward pass, and update the weights
        loss.backward()
        optimizer.step()

    # Store loss value on training
    loss_list.append(loss.data)

    # Calculate accuracy and store on training
    accuracy = 100*correct/total
    accuracy_list.append(accuracy)

    print(f"Epoch : {epoch}     "
          f"Training Accuracy : {accuracy}      "
          f"Training Loss : {loss.data.item()}     ")


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
# TEMP: loss and accuracy plots
plt.plot(range(len(loss_list)), loss_list)
plt.plot(range(len(accuracy_list)), accuracy_list)

# Prediction on testing data and preparing submission file
pred_out = model.to('cpu')(test_images)
y_pred = list(np.asarray(torch.max(pred_out.data, dim=1)[1]))
prepare_submission_file(np.asarray(y_pred))