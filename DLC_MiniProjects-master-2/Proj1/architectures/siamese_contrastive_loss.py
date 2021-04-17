# Siamese networks in pytorch: https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/


import torch
from torch import nn
from torch.nn import functional as F


import sys
sys.path.append('..') # add parent path to $PYTHONPATH so that one can import modules from parent directory
import modules.dlc_practical_prologue as prologue # import prologue

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    copied from https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class Siamese(nn.Module):
    """
    A module that trains and predicts the digits of pairs of images.

    The actual prediction for the binary target is done in compute_accuracy().
    """
    def __init__(self, use_batch_normalization):
        """
        Initializes the model, along with the weights in its 2 convolutional layers and 2 fully connected layers.
        """
        super(Siamese, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

        self.use_batch_normalization = use_batch_normalization

        if use_batch_normalization:
            self.bn1 = nn.BatchNorm2d(num_features=32)
            self.bn2 = nn.BatchNorm2d(num_features=64)
            self.bn3 = nn.BatchNorm1d(num_features=128)
        
    def forward_once(self, x):
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
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def train_model(model, train_input, train_classes,train_target, mini_batch_size):
    """
    Trains the given model for 25 epochs.

    Parameters
    -------
    model: Siamese
        The model to train
    train_input: tensor of dimension (N, 2, 14, 14)
        The training samples
    train_classes: tensor of dimension N
        The digit labels of training set (0 to 9)
    mini_batch_size: int
        The number of samples per batch

    Returns
    -------
    model
        The trained model
    """
    lr, epochs = 0.001, 25
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    auxcriterion = ContrastiveLoss()

    print("Training siamese model...")
    print("Epoch\tLoss")
    for epoch in range(0, 25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            digit_1, digit_2 = model(
                train_input[:, 0].narrow(0, b, mini_batch_size).unsqueeze(1),
                train_input[:, 1].narrow(0, b, mini_batch_size).unsqueeze(1)
            )
            loss_1 = criterion(digit_1, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_2 = criterion(digit_2, train_classes[:,1].narrow(0, b, mini_batch_size))
            auxloss = auxcriterion(torch.sigmoid(digit_1), torch.sigmoid(digit_2), train_target.narrow(0, b, mini_batch_size))
            loss = loss_1+loss_2+auxloss
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
    model: Siamese
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
        out_1, out_2 = model(
            input_data[:, 0].narrow(0, b, mini_batch_size).unsqueeze(1),
            input_data[:, 1].narrow(0, b, mini_batch_size).unsqueeze(1)
        )
        digits_1 = out_1.argmax(dim=1)
        digits_2 = out_2.argmax(dim=1)

        for i in range(mini_batch_size):
            if int(digits_1[i] <= digits_2[i]) != input_target[b + i]:
                nb_errors += 1

    accuracy = 100.0 * (N - nb_errors) / N
    return accuracy


def name():
    return "Siamese contrastive"


def main(use_batch_normalization=True, use_dropout=False):

    nb_samples = 1000
    mini_batch_size = 100

    nb_models = 10

    test_accuracies = []
    training_accuracies=[]
    for seed in range(nb_models):
        print("Model number", seed + 1)
        torch.manual_seed(seed)

        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb_samples)

        mean, std = train_input.mean(), train_input.std()

        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        # Get a random permutation of samples
        indices = torch.randperm(nb_samples)
        train_input = train_input[indices]
        train_target = train_target[indices]
        train_classes = train_classes[indices]

        model = Siamese(use_batch_normalization)

        train_model(model, train_input, train_classes, train_target, mini_batch_size)

        training_accuracy = compute_accuracy(model, train_input, train_target, mini_batch_size)
        print("Model accuracy on training set:", training_accuracy, "%")

        test_accuracy = compute_accuracy(model, test_input, test_target, mini_batch_size)
        print("Model accuracy on test set:", test_accuracy, "%")

        training_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        pytorch_total_params = sum(p.numel() for p in model.parameters())

        print("\n\n")

    print("Average siamese test accuracy:", torch.tensor(test_accuracies).mean().item(), "%")

    return test_accuracies,training_accuracies,pytorch_total_params

if __name__ == "__main__":
    main()
