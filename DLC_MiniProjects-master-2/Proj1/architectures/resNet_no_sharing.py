import torch
from torch import nn
from torch.nn import functional as F
import time

import sys
sys.path.append('..') # add parent path to $PYTHONPATH so that one can import modules from parent directory
import modules.dlc_practical_prologue as prologue # import prologue


class ResBlock(nn.Module):
    """
     Residual Block that is repeated in the Network
    """
    def __init__(self, nb_channels, kernel_size, use_batch_normalization):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
        padding = (kernel_size-1)//2)
        
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
        padding = (kernel_size-1)//2)
        
        self.use_batch_normalization = use_batch_normalization
        if self.use_batch_normalization:
            self.bn1 = nn.BatchNorm2d(nb_channels)
            self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y=self.conv1(x)
        if self.use_batch_normalization:
            y = self.bn1(y)

        y = F.relu(y)
        y = self.conv2(y)
        if self.use_batch_normalization:
            y = self.bn2(y)
        y += x
        y = F.relu(y)
        return y

class ResNet(nn.Module):
    """
    This ResNet molecule is inspired by the example in the course for Residual networks
    """
    def __init__(self, nb_channels, kernel_size, nb_blocks, use_batch_normalization, use_dropout):
        super(ResNet, self).__init__()
        self.conv0_d1 = nn.Conv2d(1, nb_channels, kernel_size = 1)
        self.conv0_d2 = nn.Conv2d(1, nb_channels, kernel_size = 1)
        self.resblocks_d1 = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size, use_batch_normalization) for _ in range(nb_blocks))
        )
        self.resblocks_d2 = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size, use_batch_normalization) for _ in range(nb_blocks))
        )
        self.avg_d1 = nn.AvgPool2d(kernel_size = 14)
        self.avg_d2 = nn.AvgPool2d(kernel_size = 14)
        self.fc_d1 = nn.Linear(nb_channels, 10)
        self.fc_d2 = nn.Linear(nb_channels, 10)
        self.use_dropout=use_dropout

        if self.use_dropout:
            self.dropout_d1=nn.Dropout()
            self.dropout_d2=nn.Dropout()

    def forward_digit1(self, x):
        x = F.relu(self.conv0_d1(x))
        x = self.resblocks_d1(x)
        x = F.relu(self.avg_d1(x))
        x = x.view(x.size(0), -1)
        x = self.fc_d1(x)
        if self.use_dropout:
            x = self.dropout_d1(x)
        return x

    def forward_digit2(self, x):
        x = F.relu(self.conv0_d2(x))
        x = self.resblocks_d2(x)
        x = F.relu(self.avg_d2(x))
        x = x.view(x.size(0), -1)
        x = self.fc_d2(x)
        if self.use_dropout:
            x = self.dropout_d2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_digit1(input1)
        output2 = self.forward_digit2(input2)
        return output1, output2
    




def train_model(model, train_input, train_classes,train_target, mini_batch_size):
    """
    Trains the ResNet model for 25 epochs.

    Parameters
    -------
    model: ResNet
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

    model.train()
    print("Training ResNet model...")
    print("Epoch\tLoss")
    for epoch in range(0, epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            digit_1, digit_2= model(
                train_input[:, 0].narrow(0, b, mini_batch_size).unsqueeze(1),
                train_input[:, 1].narrow(0, b, mini_batch_size).unsqueeze(1), 
            )
            loss_1 = criterion(digit_1, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_2 = criterion(digit_2, train_classes[:,1].narrow(0, b, mini_batch_size))
            loss = loss_1+loss_2
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
    model: ResNet
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
    model.eval()
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
    return "ResNet without weight sharing"


def main(use_batch_normalization=True, use_dropout=False):
    nb_samples = 1000
    mini_batch_size = 100
    nb_models = 10
    
    test_accuracies = []
    training_accuracies = []
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

        model = ResNet(16, 3, 5, use_batch_normalization, use_dropout)

        train_model(model, train_input, train_classes, train_target, mini_batch_size)

        training_accuracy_digit = compute_accuracy(model, train_input, train_target, mini_batch_size)
        print("Model accuracy on training set (digits): ", training_accuracy_digit, "%")


        test_accuracy  = compute_accuracy(model, test_input, test_target, mini_batch_size)
        print("Model accuracy on test set (digits) :", test_accuracy, "%")

        test_accuracies.append(test_accuracy)
        training_accuracies.append(training_accuracy_digit)
        pytorch_total_params = sum(p.numel() for p in model.parameters())



        print("\n\n")

    print(f"Average siamese test accuracy digits: {torch.tensor(test_accuracies).mean().item():.2f} % +- {torch.tensor(test_accuracies).std().item():.2f}")
    return test_accuracies,training_accuracies, pytorch_total_params

if __name__ == "__main__":
    main()
