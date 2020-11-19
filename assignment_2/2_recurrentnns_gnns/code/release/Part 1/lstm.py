"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import defaultdict
import torch
class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        # self.init_gate('g')
        self.params = nn.ParameterDict()#defaultdict(nn.Parameter)
        self.init_gate('g')
        self.init_gate('i')
        self.init_gate('f')
        self.init_gate('o')
        # print(self.params)

        self.params['W_ph'] = nn.Parameter(torch.empty(self.hidden_dim,self.num_classes))
        nn.init.kaiming_normal_(self.params['W_ph'])
        self.params['b_p'] = nn.Parameter(torch.zeros(1,self.num_classes))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)
        self.soft = nn.Softmax()
        # raise NotImplementedError
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
            out = x[:, t, :] @ self.params['W_'+name+'x'] + h_t @ self.params['W_'+name+'h'] + self.params['b_'+name]
            if 'g' in name:
                out = self.sigmoid(out)
            else:
                out = self.tanh(out)
            return out
        
        h_t = torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.batch_size,self.hidden_dim).to(self.device)
        
        for t in range(self.seq_length):
            g = frw_block(x,t,'g')
            i = frw_block(x,t,'i')
            f = frw_block(x,t,'f')
            o = frw_block(x,t,'o')    
            c_t = g*i + c_t*f
            h_t = self.tanh(c_t) * o 
        p = h_t @ self.params['W_ph'] + self.params["b_p"]
        y_hat = self.soft(p)
        return y_hat
        
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
