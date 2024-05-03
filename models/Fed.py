#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import time



def FedAvg_encrypted(weights):
    # Create a copy of the first encrypted weight tensor
    print("___________________________________________________________________________", weights)
    # h22 = time.time()
    w_avg = weights[0].copy()
    # h33 = time.time()
    # print("the time of one copy 1 ++++++++++++++++++++++++++ in ms {:f}".format((h33-h22)*1000)  )
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++_", w_avg)
    # h000=time.time()
    # Iterate over the keys of the weight tensor
    for key in w_avg.keys():
        # Initialize a sum tensor
        # h00= time.time()
        sum_encrypted = w_avg[key]
        # h11 = time.time()
        # print("the time of one copy 2 ++++++++++++++++++++++++++ in ms {:f}".format((h11-h00)*1000)  )

        # Accumulate the encrypted weights from all parties
        y=0
        for i in range(1, len(weights)):
            # h0= time.time()
            sum_encrypted += weights[i][key]
            # h1 = time.time()
            # print("the time of one addition ++++++++++++++++++++++++++ in ms {:f}".format((h1-h0)*1000)  )

        # Divide the sum by the number of parties
        num_parties = len(weights)
        # h3=time.time()
        w_avg[key] = sum_encrypted * (1.0 / num_parties)
        # h4 = time.time()
        # print("the time of one multiplicaiton ++++++++++++++++++++++++++ in ms {:f}".format((h4-h3)*1000)  )
        

    # h111 = time.time()
    # print("the time of one allllll ++++++++++++++++++++++++++ in ms {:f}".format((h111-h000)*1000)  )

    return w_avg

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    totalSize = 2
    w_avg = copy.deepcopy(w[0])
    size = 2
    size_array = np.array(size)
    size = torch.tensor(size, dtype=torch.float32)
    #print(">>>>>ii44i>>>>>",size )
    for k in w_avg.keys():
        #rint(">>>>>iii>>>>>",w_avg[k] )
        w_avg[k] = w[0][k]*size
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg
