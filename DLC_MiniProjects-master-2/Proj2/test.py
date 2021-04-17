import torch
import math

from framework.layers import Linear, Sequential
from framework.activations import ReLU, Sigmoid, Tanh, SELU
from framework.losses import LossMSE
from framework.optimizers import SGD

# if called with test.py --plots make plot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--plots", help="print performance plots",
                    action="store_true")
args = parser.parse_args()
if args.plots:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-white")

# very important
torch.set_grad_enabled(False)


def generate_set(size):
    """ Generates set of 1000 points sampled uniformly in [0, 1]^2 , each with a label
        0 if outside the disk of radius 1/√2π centered at (0.5, 0.5) and 1 otherwise.
    """
    input = torch.Tensor(size, 2).uniform_(0, 1)
    target = input.sub(0.5).pow(2).sum(dim=1).sub(1 / (2 * math.pi)).mul(-1).sign().add(1).div(2).long()
    return input, target


def conv_to_one_hot(labels):
    """convert to one hot for the two output layers and computation of the loss """
    one_hot = torch.Tensor(labels.shape[0], 2).zero_()
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return one_hot


def compute_nb_errors(out, target):
    """compare the argmax of the one hot encoded output and the targets"""
    return torch.eq(torch.argmax(out, dim=1), target).sum()


if __name__ == "__main__":
    
    torch.manual_seed(1)

    # Generate Sets
    input, target = generate_set(size=1000)
    input_test, target_test = generate_set(size=1000)

    target_labels = conv_to_one_hot(target)

    # Parameters
    num_epochs = 1000
    learning_rate = 2e-4
    batch_size = 100

    # Set up model using sequential
    # -> two input units, two output units, three hidden layers of 25 units
    model = Sequential(
        Linear(in_dim=2, out_dim=25),
        Tanh(),
        Linear(in_dim=25, out_dim=25),
        ReLU(),
        Linear(in_dim=25, out_dim=25),
        SELU(),
        Linear(in_dim=25, out_dim=2),
        Sigmoid()
    )
    
    # Specify optimizer and criterion
    optimizer = SGD(model)
    criterion = LossMSE()

    log_of_loss = []

    for epoch in range(1, num_epochs+1):
        # set gradient to 0 for each epoch
        model.zero_grad()

        loss_sum = 0
        for batch in range(0, input.shape[0], batch_size):
            # forward pass
            out = model.forward(input.narrow(0, batch, batch_size))
            loss = criterion.forward(out, target_labels.narrow(0, batch, batch_size))
            loss_sum += loss.item()

            # compute gradient of the loss
            gradwrtoutput = criterion.backward()
            # backward pass
            model.backward(gradwrtoutput)

            # Update weights
            optimizer.step(lr=learning_rate)

        if epoch % 200 == 1:
            print(f"Epoch #{epoch:3} loss: {loss_sum:.4f}", end="\n")
        else: 
            print(f"Epoch #{epoch:3} loss: {loss_sum:.4f}", end="\r")

        log_of_loss.append(loss_sum)

    # Display output
    print("\n"+"-"*50,  end="\n")
    print('Accuracy on training set:')
    after_training = model.forward(input)
    print(compute_nb_errors(after_training, target).item()/target.size()[0])

    print('Accuracy on test set:')
    after_training = model.forward(input_test)
    print(compute_nb_errors(after_training, target_test).item()/target_test.size()[0])

    # make plot if called with test.py --plots
    if args.plots:
        fig, ax = plt.subplots(1)
        ax.plot(log_of_loss, label="loss")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.savefig("loss.pdf")
