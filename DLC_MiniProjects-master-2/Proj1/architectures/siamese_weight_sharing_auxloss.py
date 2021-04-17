#First version of a Siamese CNN for MNIST pair set classification and comparison.
#Architecture used:
#-Two convolutional layers with 32 and 64 units (feature extraction).
#-Two fully connected layers with 256 and 128 units (classification).
#-Two max_pool2d layers with kernel_size 2.
#-CrossEntropyLoss method for the loss function.
#-Adam optimizer with step size eta=1e-2
#Optional perfomance improvements:
#-Use Batch normalization.
#-Use Unit Dropout.
#-Use residual networks.
#-Use momentum in Adam(?)

# Siamese networks in pytorch: https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/


import torch
from torch import nn
from torch.nn import functional as F


import sys
sys.path.append('..') # add parent path to $PYTHONPATH so that one can import modules from parent directory
import modules.dlc_practical_prologue as prologue # import prologue


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
        #training on train_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        # training on train_targets
        self.ob1_conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.ob1_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2)
        self.ob1_fc1 = nn.Linear(1024, 256)
        self.ob1_fc2 = nn.Linear(256, 2)

        self.use_batch_normalization = use_batch_normalization

        if use_batch_normalization:
            self.bn1 = nn.BatchNorm2d(num_features=32)
            self.bn2 = nn.BatchNorm2d(num_features=64)
            self.bn3 = nn.BatchNorm1d(num_features=128)

            self.ob1_bn1 = nn.BatchNorm2d(num_features=32)
            self.ob1_bn2 = nn.BatchNorm2d(num_features=64)
            self.ob1_bn3 = nn.BatchNorm1d(num_features=256)

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

    def forward_target(self, x):
        # First layer
        x = self.ob1_conv1(x)
        if self.use_batch_normalization:
            x = self.ob1_bn1(x)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

        # Second layer
        x = self.ob1_conv2(x)
        if self.use_batch_normalization:
            x = self.ob1_bn2(x)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

        x = x.reshape(x.size(0), -1).view(-1, 1024)

        # Third layer
        x = self.ob1_fc1(x)
        if self.use_batch_normalization:
            x = self.ob1_bn3(x)
        x = F.relu(x)

        # Final layer
        x = torch.sigmoid(self.ob1_fc2(x))
        return x
    
    def forward(self, input1, input2, complete_train_input):
        digit1 = self.forward_once(input1)
        digit2 = self.forward_once(input2)
        target = self.forward_target(complete_train_input)

        return digit1, digit2, target


def train_model(model, train_input, train_classes, train_target, mini_batch_size):
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
    train_target: tensor of dimension N
        The boolean target of training set
    mini_batch_size: int
        The number of samples per batch

    Returns
    -------
    model
        The trained model
    """

    lr, epochs = 0.001, 25
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    print("Training siamese model...")
    print("Epoch\tLoss")
    for epoch in range(0, 25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            digit_1, digit_2, score = model(
                train_input[:, 0].narrow(0, b, mini_batch_size).unsqueeze(1),
                train_input[:, 1].narrow(0, b, mini_batch_size).unsqueeze(1), 
                train_input.narrow(0, b, mini_batch_size)
            )
            loss_1 = criterion1(digit_1, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_2 = criterion1(digit_2, train_classes[:,1].narrow(0, b, mini_batch_size))
            auxloss = criterion2(score, train_target.narrow(0, b, mini_batch_size))
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

    nb_errors_digits, nb_errors_targets = 0,0
    for b in range(0, N, mini_batch_size):
        out_1, out_2, out_3 = model(
            input_data[:, 0].narrow(0, b, mini_batch_size).unsqueeze(1),
            input_data[:, 1].narrow(0, b, mini_batch_size).unsqueeze(1),
            input_data.narrow(0,b,mini_batch_size)
        )
        digits_1 = out_1.argmax(dim=1)
        digits_2 = out_2.argmax(dim=1)
        score=out_3.argmax(dim=1)

        for i in range(mini_batch_size):
            if int(digits_1[i] <= digits_2[i]) != input_target[b + i]:
                nb_errors_digits += 1
            if score[i] != input_target[b + i]:
                nb_errors_targets += 1

    accuracy_digits = 100.0 * (N - nb_errors_digits) / N
    accuracy_targets = 100.0 * (N - nb_errors_targets) / N
    
    return accuracy_digits, accuracy_targets


def name():
    return "Siamese auxloss"


def main(use_batch_normalization=True, use_dropout=False):

    nb_samples = 1000
    mini_batch_size = 100

    nb_models = 10

    test_accuracies_digit, test_accuracies_target = [], []
    training_accuracies=[]
    test_accuracies=[]
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

        training_accuracy_digit, training_accuracy_target = compute_accuracy(model, train_input, train_target, mini_batch_size)
        print("Model accuracy on training set (digits): ", training_accuracy_digit, "%")
        print("Model accuracy on training set (targets):", training_accuracy_target, "%")

        test_accuracy_digit,test_accuracy_target  = compute_accuracy(model, test_input, test_target, mini_batch_size)
        print("Model accuracy on test set (digits) :", test_accuracy_digit, "%")
        print("Model accuracy on test set (targets):", test_accuracy_target, "%")


        training_accuracies.append(training_accuracy_digit)
        test_accuracies.append(test_accuracy_digit)
        pytorch_total_params = sum(p.numel() for p in model.parameters())

        print("\n\n")

    print(f"Average siamese test accuracy digits: {torch.tensor(test_accuracies_digit).mean().item():.2f} % +- {torch.tensor(test_accuracies_digit).std().item():.2f}")
   # print(f"Average siamese test accuracy targets: {torch.tensor(test_accuracies_target).mean().item():.2f} % +- {torch.tensor(test_accuracies_target).std().item():.2f}")

    return test_accuracies,training_accuracies,pytorch_total_params


if __name__ == "__main__":
    main()
