'''
    Project 1 by 
        Adam Misik
        Lua Streit
        Simon Dürr
    
    This is a wrapper that will call all different architectures that were tested and will print performance statistics. 

    call with `python test.py`
'''


from architectures import siamese_target
from architectures import siamese_no_sharing
from architectures import siamese_class
from architectures import simpleNet
from architectures import siamese_contrastive_loss as contrastive
from architectures import siamese_weight_sharing_auxloss as auxloss
from architectures import resNet
from architectures import resNet_no_sharing


import torch
import time
import sys
import os


def block_print():
    """
    Helper function that temporarily removes all print statements (to keep the console output clean).
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Helper function that brings back the possibility of printing characters to the console.
    """
    sys.stdout = sys.__stdout__


def measure_stats(arch, block, args):
    print("/----------------TESTING------------------\\")
    print("|", arch.name(), "with" if use_batch_normalization else "without", "batch normalization")
    print("|", arch.name(), "with" if use_dropout else "without", "dropout")
    # if you want the print statements in model training to apper, comment this line
    block_print()

    start = time.perf_counter()
    test_accuracies,training_accuracies,parameters = block(*args)
    end = time.perf_counter()

    enable_print()

    mean_test = torch.tensor(test_accuracies).mean().item()
    std_test = torch.tensor(test_accuracies).std().item()
    mean_train = torch.tensor(training_accuracies).mean().item()
    std_train = torch.tensor(training_accuracies).std().item()
    train_time = (end - start) / len(test_accuracies)
    
    #params.append(parameters)
    #train_time_sum.append(train_time)
    #mean_test_accu.append(mean_test)
    #mean_train_accu.append(mean_train)


    print("| Test Accuracy mean: %.2f %% " % mean_test)
    print("| Test Accuracy std: %.2f %% " % std_test)
    print("| Training Accuracy mean: %.2f %% " % mean_train)
    print("| Training Accuracy std: %.2f %% " % std_train)
    print("| Mean model training time: %.2f s" % train_time)
    print(f"| Parameters: {parameters}")
    print("\\-----------------DONE--------------------/\n\n")
    return


if __name__ == "__main__":
    """
    On execution of the test.py file all different architectures will be called and will print their respective performance statistics.
    """

    global_start = time.perf_counter()
    
    
    for arch in [resNet, resNet_no_sharing]:
        for use_batch_normalization in [True, False]:
            for use_dropout in [True, False]:
                measure_stats(arch, arch.main, (use_batch_normalization, use_dropout))
    
    use_dropout=False
    for arch in [auxloss, contrastive, siamese_class, siamese_target,siamese_no_sharing, simpleNet]:
        for use_batch_normalization in [True, False]:
            measure_stats(arch, arch.main, (use_batch_normalization,use_dropout))
            

    global_end = time.perf_counter()

    print("All architectures were successfully trained and tested in %.2f s!" % (global_end - global_start))

