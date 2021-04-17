import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import sys

sys.path.append('..')  # add parent path to $PYTHONPATH so that one can import modules from parent directory
import modules.dlc_practical_prologue as prologue  # import prologue

class SimpleNet(nn.Module):
    """
    Setup of the module for the simple convolutional Net. 

    This module applies two convolutions and treats the two digits as two channel images. 
    It is expected to achieve relatively modest accuracy on the test set. 

    The module applies two convolutions, where after each convolution the ReLU activiation and then Max pooling filters are applied. 
    Then two fully connected layers are used to narrow down to two output units. 

    The two output units are interpreted that first number is bigger or second number is bigger. 
    """

    def __init__(self, use_batch_normalization):
        """
          initialize the layers of the net            
        """
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)

        self.use_batch_normalization = use_batch_normalization
        if use_batch_normalization:
            self.bn1 = nn.BatchNorm2d(num_features=32)
            self.bn2 = nn.BatchNorm2d(num_features=64)
            self.bn3 = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        """Forward function 
        
        Returns
        -------
        tensor
            dimension of 2x1, tensor[0]>tensor[1] means objective is fullfilled
        """
        # First layer
        x = self.conv1(x)
        if self.use_batch_normalization:
            x = self.bn1(x)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

        # Second layer
        x = self.conv2(x)
        if self.use_batch_normalization:
            x = self.bn2(x)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

        x = x.reshape(x.size(0), -1)
        x = x.view(-1, 4 * 4 * 64)

        # Third layer
        x = self.fc1(x)
        if self.use_batch_normalization:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Final layer
        x = torch.sigmoid(self.fc2(x))
        return x


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """
    Computes error rate

    Parameters
    ----------
    model : object
        Simple Net
    data_input : tensor
        training input Nx2 classes from the Net
    data_target : tensor
        true classes of the input data set Nx2
    mini_batch_size : int
        Mini batch size

    Returns
    -------
    int
        number of errors
    """
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def train_model(model, train_input, train_target, test_input, test_target, device, nb_epochs, mini_batch_size):
    """
    Train the simple Net

    Parameters
    ----------
    model : class
        simpleNet class from above
    train_input : tensor
        Nx2x14x14 training images from MNIST
    train_target : tensor
        Nx2 classes for train_input
    test_input : tensor
        Nx2x14x14 testing images from MNIST
    test_target : tensor
        Nx2 classes for test_input
    device : misc
        cuda or no cuda
    nb_epochs : int
        number of training epochs
    mini_batch_size : int
        number of mini batches

    """

    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()
    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Vectors to accumulate the losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []

    for e in range(nb_epochs):

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss.append(loss.item())
        print(f'\nEpoch : {e}, Loss: {loss.item()}')

        model.eval()
        training_accuracy.append(
            100 - compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)
        print(f'Accuracy on training set: {training_accuracy[-1]}')
        validation_accuracy.append( \
            100 - compute_nb_errors(model, test_input, test_target, mini_batch_size) / (test_input.size(0)) * 100)
        print(f'Accuracy on test set: {validation_accuracy[-1]}')

        model.train()

    return model, training_loss, training_accuracy, validation_accuracy


def main(use_batch_normalization=True, use_dropout=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("using cuda!")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    nb_epochs = 25
    mini_batch_size = 100

    nb_models = 10

    test_accuracies = []
    training_accuracies=[]
    for seed in range(nb_models):
        torch.manual_seed(seed)

        model = SimpleNet(use_batch_normalization)
        # move model to correct device
        model.to(device)

        # create training and testing data
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            1000)

        # normalize set
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        # move to device
        train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(
            device)
        test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)

        # start the training
        model, training_loss, training_accuracy, validation_accuracy = train_model(model, \
                                                                                   train_input, \
                                                                                   train_target, \
                                                                                   test_input, \
                                                                                   test_target, \
                                                                                   device, \
                                                                                   nb_epochs, \
                                                                                   mini_batch_size)

        training_accuracies.append(training_accuracy)
        test_accuracies.append(validation_accuracy)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        
    return test_accuracies,training_accuracy,pytorch_total_params


def name():
    return "Simple Net"


if __name__ == "__main__":
    main()

