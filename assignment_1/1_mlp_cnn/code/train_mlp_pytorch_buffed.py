"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch_buffed import MLP
import cifar10_utils

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# Default constants

from torch.optim.lr_scheduler import StepLR
DNN_HIDDEN_UNITS_DEFAULT = '500,300,100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 14000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    predictions_labels = np.argmax(predictions,axis=1)
    labels = np.argmax(targets,axis=1)
    n_labels = labels.shape[0]
    # correct = (predictions_labels == labels).sum().item()
    counts = np.sum([1  for i in range(n_labels) if predictions_labels[i] == labels[i]  ])
    accuracy = counts / n_labels * 100
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():

    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    # neg_slope = FLAGS.neg_slope
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # transforms.Compose(
    #     # [transforms.RandomResizedCrop(224),
    #     #  transforms.RandomHorizontalFlip(),
    #     #  transforms.ToTensor(),
    #     [transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    data = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x,labels = data['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(np.size(x,0),-1)
    model = MLP(x.shape[1],dnn_hidden_units,10)
    model.to(device)
    crossE = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate,momentum=0.7, nesterov=True)
    # opt = torch.optim.Adam(model.parameters(),lr=FLAGS.learning_rate)

    losses = []
    accs = []
    val_losses = []
    val_accs =[]
    best_acc = 0 
    
    scheduler = StepLR(opt, step_size=10, gamma=0.9)
    for it in tqdm(range(FLAGS.max_steps)):
        # for batch in range(10):
        if it % 100 ==0 :
            scheduler.step()
        model.train()
        x = torch.from_numpy(x).to(device)
        opt.zero_grad()
        
        preds = model(x)
        # print(labels)
        label_idxs = torch.argmax(torch.from_numpy(labels),dim= 1).long().to(device)
        print('here',label_idxs.shape,label_idxs)
        print('here',preds.shape,preds)
        loss = crossE(preds,label_idxs)
        # print(loss)
        # exit()
        loss.backward()
        opt.step()
        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('accs/train',accuracy(preds.detach().cpu(),labels), it)

        if it % FLAGS.eval_freq == 0 :
            model.eval()
            # losses.append(loss.item())
            # accs.append(accuracy(preds.detach().cpu(),labels))
        
            
            x      = data['test'].images
            labels = data['test'].labels
            x = x.reshape(np.size(x,0),-1)
            x = torch.from_numpy(x).to(device)
            label_idxs = torch.argmax(torch.from_numpy(labels),dim= 1).long().to(device)
        
            preds = model(x)
            val_loss = crossE(preds,label_idxs)
            # val_losses.append(val_loss.item())
            # val_accs.append()
            curr_acc = accuracy(preds.detach().cpu(),labels)
            writer.add_scalar('accs/val',curr_acc, it)
            writer.add_scalar('Loss/val',val_loss.item(), it)

            if curr_acc > best_acc:
                best_acc = curr_acc
                print('best model so far with validation accuracy of ', curr_acc)
        x,labels = data['train'].next_batch(BATCH_SIZE_DEFAULT)
        x = x.reshape(np.size(x,0),-1)
        
    return model  
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    model = train()
    # losses = np.squeeze(np.array(losses))
    # print(losses.shape)
    # val_losses = np.squeeze(np.array(val_losses))
    # accs = np.squeeze(np.array(accs))
    # val_accs = np.squeeze(np.array(val_accs))
    # train_x_axis = range(len(losses))
    # val_x_axis = range(len(val_losses))

    # plt.figure(1)
    # plt.plot(val_x_axis,val_accs,label='Validation accuracy')
    
    # # plt.plot(train_x_axis,accs,label='Train accuracy')
    # plt.legend()
    # plt.figure(2)
    # plt.plot(train_x_axis,losses,label='Train loss')
    
    # # plt.plot(val_x_axis,val_losses,label='Validation loss')
    # plt.legend()

    # plt.figure(3)
    # # plt.plot(val_x_axis,val_accs,label='Validation accuracy')
    
    # plt.plot(train_x_axis,accs,label='Train accuracy')
    # plt.legend()
    # plt.figure(4)
    # # plt.plot(train_x_axis,losses,label='Train loss')
    
    # plt.plot(val_x_axis,val_losses,label='Validation loss')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
