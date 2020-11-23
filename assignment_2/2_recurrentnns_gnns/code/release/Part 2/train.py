# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
writer = SummaryWriter('runs/s2slstm')
###############################################################################
def my_accuracy(preds,trgs):
    preds = preds.argmax(dim=1)
    acc = (preds == trgs).float().mean()
    # acc = [1 if preds[i] == trgs[i] else 0 for i in range(len(preds)) ].sum().mean()
    return acc
def gen_sentence(model,dataset,length,temp = 0):
    char = torch.randint(0,dataset.vocab_size,(1,1))
    sentence = []
    h = None
    model.eval()
    for l in range(length):
        # model.zero_grad()

        preds,h = model.forward(char,h)
        # print(preds)
        # print(preds.argmax())
        if temp == 0:
            char[0,0] = preds.squeeze().argmax()
        else:
            pd = preds.squeeze()/temp
            pd = torch.softmax(pd,dim=0)
            char[0,0] = torch.multinomial(pd,1)

        sentence.append(char.item())
    # print(dataset.convert_to_string(sentence))
    # exit()
        # char[0,0] = 
    return dataset.convert_to_string(sentence)
def train(config):

    # Initialize the device which to run the model on
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    book2_path = 'assets/book_EN_grimms_fairy_tails.txt'
    dataset = TextDataset(book2_path,seq_length=config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size, seq_length=config.seq_length, vocabulary_size=dataset.vocab_size)  # FIXME
    model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    # print(model.parameters)
    optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate)  # FIXME

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        model.train()
        # print(t?orch.stack(batch_inputs))
        # exit()?
        # meh = [print(dataset.convert_to_string(list(x))) for x in batch_inputs]
        # batch_inputs[0] = batch_inputs[0].numpy()

        # batch_targets[0] = batch_targets[0].numpy()
        # print(len(batch_inputs))
        # print(len(batch_targets))
        # print(list(batch_inputs[ 0]))

        # print(dataset.convert_to_string(list(batch_inputs[0])))
        # exit()
        model.zero_grad()
        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets,dim=1).to(device)
        preds,(_ ,_)= model.forward(batch_inputs)
        # print(preds.shape)
        preds = preds.permute(1,2,0)
        # exit()
        # print(preds.shape)
        loss = criterion(preds,batch_targets)   # fixme
        accuracy = my_accuracy(preds,batch_targets)  # fixme
        writer.add_scalar('Loss',loss.item(),step)
        writer.add_scalar('Accuracy',accuracy*100,step)
        loss.backward()
        # print(loss)
        # exit()
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                     config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            sentence = gen_sentence(model,dataset,30,temp = 0)
            print(sentence)
            pass

        if step == config.train_steps:
            torch.save(model.state_dict(), config.save_model)
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
