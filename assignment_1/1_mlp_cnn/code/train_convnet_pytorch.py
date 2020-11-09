"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    counts = np.sum([1  for i in range(n_labels) if predictions_labels[i] == labels[i]  ])
    accuracy = counts / n_labels
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    data = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x,labels = data['train'].next_batch(FLAGS.batch_size)
    
    x = x.reshape(FLAGS.batch_size,3,32,32)
    print(x.shape)
    model = ConvNet(3,10)
    model.to(device)
    crossE = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate,momentum=0.7, nesterov=True)
    # opt = torch.optim.Adam(model.parameters(),lr=FLAGS.learning_rate)

    losses = []
    accs = []
    val_losses = []
    val_accs =[]
    best_acc = 0 
    print('training...')
    # scheduler = StepLR(opt, step_size=10, gamma=0.9)
    for it in tqdm(range(FLAGS.max_steps)):
        
        # for batch in range(10):
        # if it % 100 ==0 :
            # scheduler.step()
        model.train()
        x = torch.from_numpy(x).to(device)
        opt.zero_grad()
        
        preds = model(x)
        # print(labels)
        label_idxs = torch.argmax(torch.from_numpy(labels),dim= 1).long().to(device)
        # print('here',label_idxs)
        # print('here',label_idxs.shape,label_idxs)
        # print('here',preds.shape,preds)
        loss = crossE(preds,label_idxs)
        # print(loss)
        # exit()
        loss.backward()
        opt.step()
        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('accs/train',accuracy(preds.detach().cpu(),labels), it)

        if it % FLAGS.eval_freq == 0 :
            print('testing...')
            model.eval()
            # losses.append(loss.item())
            # accs.append(accuracy(preds.detach().cpu(),labels))
            # print(data['test'].num_exampl)
            
            curr_acc_t = 0
            curr_l_t = 0
            n_iters_test = int(data['test'].num_examples/FLAGS.batch_size)
            for t_it in tqdm(range(n_iters_test)):
                
            # x      = data['test'].images
            # labels = data['test'].labels
                x,labels = data['test'].next_batch(FLAGS.batch_size)
                # print(x.shape)
                # x = x.reshape(FLAGS.batch_size,3,32,32)
                
                # x = x.reshape(np.size(x,0),-1)
                x = torch.from_numpy(x).to(device)
                label_idxs = torch.argmax(torch.from_numpy(labels),dim= 1).long().to(device)
            
                preds = model(x)
                val_loss = crossE(preds,label_idxs)
                # val_losses.append(val_loss.item())
                # val_accs.append()
                curr_acc_t += accuracy(preds.detach().cpu(),labels)
                curr_l_t += val_loss.item()
            curr_acc_t = curr_acc_t / n_iters_test
            curr_l_t = curr_l_t / n_iters_test
            writer.add_scalar('accs/val',curr_acc_t, it)
            writer.add_scalar('Loss/val',curr_l_t, it)

            if curr_acc_t > best_acc:
                best_acc = curr_acc_t
                print('best model so far with validation accuracy of ', curr_acc_t)
            print('training...')
        x,labels = data['train'].next_batch(BATCH_SIZE_DEFAULT)
        x = x.reshape(FLAGS.batch_size,3,32,32)
        # x = x.reshape(np.size(x,0),-1)
        
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
    torch.save(model.state_dict(),'models/best_model.pth')


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
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
