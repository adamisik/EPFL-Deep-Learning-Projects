# First version of a Siamese CNN for MNIST pair set classification and comparison.
# Architecture used:
# -Two convolutional layers with 32 and 64 units (feature extraction).
# -Two fully connected layers with 256 and 128 units (classification).
# -Two max_pool2d layers with kernel_size 2.
# -CrossEntropyLoss method for the loss function.
# -Adam optimizer with step size eta=1e-2
# Optional perfomance improvements:
# -Use Batch normalization.
# -Use Unit Dropout.
# -Use residual networks.
# -Use momentum in Adam(?)

# Siamese networks in pytorch: https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/


import torch
from torch import nn
from torch.nn import functional as F

import sys

sys.path.append('..')  # add parent path to $PYTHONPATH so that one can import modules from parent directory
import modules.dlc_practical_prologue as prologue  # import prologue


class SiameseTarget(nn.Module):
    """
    A module that trains and predicts the boolean target pairs of images.

    The actual prediction for the binary target is done in compute_accuracy().
    """

    def __init__(self, use_batch_normalization):
        """
        Initializes the model, along with the weights in its 2 convolutional layers and 4 fully connected layers.
        """
        super(SiameseTarget, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)

        self.use_batch_normalization = use_batch_normalization
        if use_batch_normalization:
            self.bn1 = nn.BatchNorm2d(num_features=32)
            self.bn2 = nn.BatchNorm2d(num_features=64)
            self.bn3 = nn.BatchNorm1d(num_features=128)

    def common_path(self, x):
        # First layer
        x = self.conv1(x)
        if self.use_batch_normalization:
            x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2))

        # Second layer
        x = self.conv2(x)
        if self.use_batch_normalization:
            x = self.bn2(x)
        x = F.relu(x)

        x = x.view(-1, 64)

        # Third layer
        x = self.fc1(x)
        if self.use_batch_normalization:
            x = self.bn3(x)
        x = F.relu(x)

        # Final layer
        x = self.fc2(x)
        return x

    def forward(self, input):
        input_1 = input[:, 0].unsqueeze(dim=1)
        input_2 = input[:, 1].unsqueeze(dim=1)

        output1 = self.common_path(input_1)
        output2 = self.common_path(input_2)

        output = torch.cat((output1, output2), dim=1)

        output = F.relu(self.fc3(output))
        output = self.fc4(output)

        return output


def train_model(model, train_input, train_target, mini_batch_size):
    """
    Trains the given model for 25 epochs.

    Parameters
    -------
    model: SiameseTarget
        The model to train
    train_input: tensor of dimension (N, 2, 14, 14)
        The training samples
    train_target: tensor of dimension N
        The 1-0 boolean labels of the training set
    mini_batch_size: int
        The number of samples per batch

    Returns
    -------
    model
        The trained model
    """
    # TODO was this supposed to be here?
    # module.cuda()
    lr, epochs = 0.001, 25
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    print("Training siamese target model...")
    print("Epoch\tLoss")
    for epoch in range(0, 25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            out = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(out, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch + 1, sum_loss, sep="\t", end="\r")

    print()
    print("Done training!\n")

    return model


def compute_accuracy(model, input_data, input_target, mini_batch_size):
    """
    Computes the accuracy of the model on the input data.

    Parameters
    -------
    model: SiameseTarget
        The model we want to test the accuracy of
    input_data: tensor of dimension (N, 2, 14, 14)
        The data on which we want to the test the model
    input_target: tensor of dimension N
        The 1-0 boolean targets
    mini_batch_size: int
        The number of samples per batch

    Returns
    -------
    accuracy
        The percentage of correct predictions
    """
    N = input_data.shape[0]

    nb_errors = 0
    for b in range(0, N, mini_batch_size):
        out = model(input_data.narrow(0, b, mini_batch_size)).argmax(dim = 1)
        for i in range(mini_batch_size):
            if out[i] != input_target[b + i]:
                nb_errors += 1

    accuracy = 100.0 * (N - nb_errors) / N
    return accuracy


def main(use_batch_normalization=True, use_dropout=False):
    nb_samples = 1000
    mini_batch_size = 100

    test_accuracies = []
    training_accuracies=[]
    for seed in range(10):
        print("Model number", seed + 1)
        torch.manual_seed(seed)

        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            nb_samples)

        mean, std = train_input.mean(), train_input.std()

        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        # Get a random permutation of samples
        indices = torch.randperm(nb_samples)
        train_input = train_input[indices]
        train_target = train_target[indices]

        model = SiameseTarget(use_batch_normalization)

        train_model(model, train_input, train_target, mini_batch_size)

        training_accuracy = compute_accuracy(model, train_input, train_target, mini_batch_size)
        print("Model accuracy on training set:", training_accuracy, "%")

        test_accuracy = compute_accuracy(model, test_input, test_target, mini_batch_size)
        print("Model accuracy on test set:", test_accuracy, "%")

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        pytorch_total_params = sum(p.numel() for p in model.parameters())

    return test_accuracies,training_accuracies,pytorch_total_params


def name():
    ## TODO replace
    return "Siamese Target"


if __name__ == "__main__":
    main()
