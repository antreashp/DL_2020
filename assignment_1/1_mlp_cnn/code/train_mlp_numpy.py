"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 600
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
writer = SummaryWriter('runs/numpy')
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
    counts = np.sum([1  for i in range(n_labels) if predictions_labels[i] == labels[i]  ])
    accuracy = counts / n_labels
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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # module_1 = LinearModule(200, 10)
    # rand_x = np.random.random((32, 200))
    # print(module_1.forward(rand_x).shape)
    # print(module_1.backward(dout=np.ones((32, 10))))
    # exit(0)

    data = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x,labels = data['train'].next_batch(FLAGS.batch_size)

    x = x.reshape(np.size(x,0),-1)
    # print(x.shape)
    # exit(0)



    # print(x.shape)
    # print('normalizing...')
    # b = [[(i-min(l))/(max(l)-min(l)) for i in tqdm(l)] for l in x]
    # print(b)
    # exit()
    model = MLP(x.shape[1],dnn_hidden_units,10)
    crossE = CrossEntropyModule()
    losses = []
    accs = []
    val_losses = []
    val_accs =[]

    print('Training...')
    for it in tqdm(range(FLAGS.max_steps)):
        
        # model = model.train()
        # for l in model.layers:
        #     # l.grads['weights'] = 
        #     l.grads['weights'] = np.zeros_like(l.grads['weights'])
        #     l.grads['bias'] = np.zeros_like(l.grads['bias'])
        # x = np.ones_like(x)
        
        preds = model.forward(x)       #frwrd
        loss = crossE.forward(preds,labels) #criterion
        # print(loss.shape)
        losses.append(loss)
        accs.append(accuracy(preds,labels))
        # print(loss)
        d = crossE.backward(preds,labels) 
        model.backward(d)

        # print(model.out_layer.grads['weights'])4
        # print('meh')
        for l in model.layers: #update
            # print('here')
            # print('here')
            # print(l)
            l.params['weights'] = l.params['weights'] - FLAGS.learning_rate * l.grads['weights']
            l.params['bias'] = l.params['bias'] - FLAGS.learning_rate * l.grads['bias']
        model.out_layer.params['weights'] = model.out_layer.params['weights'] - FLAGS.learning_rate *model.out_layer.grads['weights']
        model.out_layer.params['bias'] = model.out_layer.params['bias'] - FLAGS.learning_rate *model.out_layer.grads['bias']
        # plot_grad_flow(model.named_parameters())
        # print(model.out_layer.grads['weights'])
        # exit()
        if it % FLAGS.eval_freq == 0:
            # model = model.eval()
            print('Testing...')
            x      = data['test'].images
            labels = data['test'].labels
            x = x.reshape(np.size(x,0),-1)
            preds = model.forward(x)
            val_loss = crossE.forward(preds,labels)
            val_losses.append(val_loss)
            val_accs.append(accuracy(preds,labels))
            if val_accs[-1] > max(val_accs):
                print('best model so far with validation accuracy of ', val_accs[-1])
            print('Training...')
        # x,labels = data['train'].next_batch(BATCH_SIZE_DEFAULT)

        # x = x.reshape(np.size(x,0),-1)
        x,labels = data['train'].next_batch(FLAGS.batch_size)
        x = x.reshape(np.size(x,0),-1)
    return model,losses,val_losses,accs,val_accs

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
    model,losses,val_losses,accs,val_accs = train()
    losses = np.squeeze(np.array(losses))
    print(losses.shape)
    val_losses = np.squeeze(np.array(val_losses))
    accs = np.squeeze(np.array(accs))
    val_accs = np.squeeze(np.array(val_accs))
    train_x_axis = range(len(losses))
    val_x_axis = range(len(val_losses))

    plt.figure(1)
    plt.plot(train_x_axis,losses,label='Train loss')
    plt.plot(train_x_axis,accs,label='Train accuracy')
    plt.legend()
    plt.figure(2)
    plt.plot(val_x_axis,val_losses,label='Validation loss')
    plt.plot(val_x_axis,val_accs,label='Validation accuracy')
    plt.legend()
    plt.show()

    # fig, axs = plt.subplots(1, 2, sharey='row')
    # print(np.array(train_x_axis).shape)
    # print(np.array(losses).shape)
    # axs[0,0].plot(list(train_x_axis),list(losses),label='Train loss')
    
    # # axs[0,0].plot(train_x_axis,accs,label='Train accuracy')
    # axs[0,0].legend()
    # # axs[0,1].plot(train_x_axis,losses,label='Validation loss')
    
    # axs[0,1].plot(val_x_axis,accs,label='Validation accuracy')
    # axs[0,1].legend()
    # plt.show()
    # plt.plot(train_x_axis,losses)    

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
