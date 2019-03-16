import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PreprocessingFindingTheLargestContour import preprocess_training_data, preprocess_test_data


def prepare_training_data(input_path, size_reduction=0, batch_size=100):

    # load training data as numpy array
#     features_np = pd.read_pickle(input_path + '/train_images.pkl')
#     features_np = features_np*(features_np > 240)     # noise cancelling
#     features_np = features_np/255  # normalization
    features_np = preprocess_training_data(input_path)
    targets_np = np.array(pd.read_csv(input_path + '/train_labels.csv'))
    targets_np = targets_np[:, -1]  # delete id column

    # reduce data size (used for debugging purposes)
    if size_reduction > 0:
        used_data = int(len(features_np)*(1-size_reduction))
        features_np = features_np[:used_data]
        targets_np = targets_np[:used_data]

    # train - validation sets split
    features_train_np, features_val_np, targets_train_np, targets_val_np = train_test_split(
        features_np, targets_np, test_size=0.20, random_state=501)

    # convert to tensor, add depth dimension as 1 (by unsqueeze)
    features_train_t = torch.unsqueeze(torch.from_numpy(features_train_np), 1)
    features_train_t = features_train_t.type('torch.FloatTensor')
    targets_train_t = torch.from_numpy(targets_train_np)
    features_val_t = torch.unsqueeze(torch.from_numpy(features_val_np), 1)
    features_val_t = features_val_t.type('torch.FloatTensor')
    targets_val_t = torch.from_numpy(targets_val_np)

    # form pytorch training and validation sets
    train = torch.utils.data.TensorDataset(features_train_t, targets_train_t)
    val = torch.utils.data.TensorDataset(features_val_t, targets_val_t)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def prepare_submission_data(input_path, batch_size=10):

    # load training data as numpy array
#     features_train_np = pd.read_pickle(input_path + '/train_images.pkl')
#     features_train_np = features_train_np*(features_train_np > 240)     # noise cancelling
#     features_train_np = features_train_np/255  # normalization
    features_train_np = preprocess_training_data(input_path)
    targets_train_np = np.array(pd.read_csv(input_path + '/train_labels.csv'))
    targets_train_np = targets_train_np[:, -1]  # delete id column

    # convert to tensor, add depth dimension as 1 (by unsqueeze)
    features_train_t = torch.unsqueeze(torch.from_numpy(features_train_np), 1)
    features_train_t = features_train_t.type('torch.FloatTensor')
    targets_train_t = torch.from_numpy(targets_train_np)

    # form pytorch training sets
    train = torch.utils.data.TensorDataset(features_train_t, targets_train_t)

    # training data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    # load testing data as numpy array and convert to tensor
#     features_test_np = pd.read_pickle(input_path + '/test_images.pkl')
#     features_test_np = features_test_np*(features_test_np > 240)     # noise cancelling
#     features_test_np = features_test_np/255  # normalization
    features_test_np = preprocess_test_data(input_path)

    # convert to tensor, add depth dimension as 1 (by unsqueeze)
    test_images = torch.unsqueeze(torch.from_numpy(features_test_np), 1)
    test_images = test_images.type('torch.FloatTensor')


    return train_loader, test_images


def prepare_submission_file(y_pred):
    df = pd.DataFrame()
    df['Id'] = range(len(y_pred))
    df['Category'] = y_pred
    output_name = "submission" + str(random.randint(0, 1000)) +".csv"
    print("The submission file is named as ", output_name)
    df.to_csv(path_or_buf="submissions/"+output_name, index=False)


def visualization(features_np, targets_np, i):
    plt.imshow(features_np[i].reshape(64, 64))
    plt.axis("off")
    plt.title(str(targets_np[i]))
    plt.show()
