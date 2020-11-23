"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        '''
            f = sig(W_fx*x_t + W_fh*c_t-1 +bf )
            i = sig(W_ix*x_t + W_ih*c_t-1 +bi )
            o = sig(W_ox*x_t + W_oh*c_t-1 +bo )
            c = sig(W_cx*x_t + bc ) * i_t +c_t-1 *f_t
            g = tanh(c_t)*o_t
            p_t = W_ph*h_t +bp
            y_hat = softmax(p_t)
        '''
        self.params = nn.ParameterDict()
        self.init_gate('f')
        self.init_gate('i')
        self.init_gate('o')
        self.init_gate('c')
        # self.init_gate('g')
        self.params['W_ph'] = nn.Parameter(torch.empty(self.hidden_dim,self.num_classes))
        nn.init.kaiming_normal_(self.params['W_ph'])
        self.params['b_p'] = nn.Parameter(torch.zeros(1,self.num_classes))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)
        self.soft = nn.Softmax()
        ########################
        # END OF YOUR CODE    #
        #######################
    def init_gate(self,gate):
        # print('here',self.parameters)
        namex = 'W_'+gate+'x'
        nameh = 'W_'+gate+'h' 
        nameb = 'b_'+gate
        self.params[namex] = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.params[nameh] = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.kaiming_normal_(self.params[namex])
        nn.init.kaiming_normal_(self.params[nameh])
        self.params[nameb] = nn.Parameter(torch.zeros(1,self.hidden_dim))
    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        def frw_block(x,t,name):            
            out = self.sigmoid(x[:, t, :] @ self.params['W_'+name+'x'] + h_t @ self.params['W_'+name+'h'] + self.params['b_'+name])
            # if 'c' not in name:
            #     out = self.sigmoid(out)
            # else:
            #     out = self.tanh(out)
            return out
        h_t = torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        
        for t in range(self.seq_length):
            f = frw_block(x,t,'f')
            i = frw_block(x,t,'i')
            o = frw_block(x,t,'o')
            # c = frw_block(x,t,'c')
            # g = frw_block(x,t,'g')
            c_t = (self.tanh(x[:, t, :] @ self.params['W_'+'c'+'x'] + self.params['b_'+'c'])) *i + c_t*f
            # c_t = g*i + c_t*f
            h_t = self.tanh(c_t) * o 
        p = h_t @ self.params['W_ph'] + self.params["b_p"]
        y_hat = self.soft(p)
        return y_hat
        ########################
        # END OF YOUR CODE    #
        #######################

