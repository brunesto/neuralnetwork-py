# basic neural network implementation using no libraries

import math
import random
import copy
from NeuralNetConfig import *


def zeros(s):
    v = []
    for i in range(0, s):
        v.append(0)
    return v


def zeros2d(s1, s2):
    v = []
    for i in range(0, s1):
        v.append(zeros(s2))
    return v




class NeuralNet:

    def __init__(self, config):

        self.config = config

        # layers
        self.ls = []
        for i in range(0, len(config.layer_sizes)):
            self.ls.append(zeros(config.layer_sizes[i]))

        # for each neuron, keep its z value (z: sum of weighted input (incl bias) before activation function)
        self.zs = []
        for i in range(0, len(config.layer_sizes)):
            self.zs.append(zeros(config.layer_sizes[i]))

        # weights
        # ws[0] are the weights to update l[1] from l[0]
        # ws[0][-1] is an extra weight which is the bias
        self.ws = []
        for li in range(1, len(config.layer_sizes)):
            self.ws.append(zeros2d(config.layer_sizes[li], config.layer_sizes[li - 1]))

        # biases
        self.bs = []
        for li in range(1, len(config.layer_sizes)):
            self.bs.append(zeros(config.layer_sizes[li]))

        # now randomly populate weights
        random.seed(self.config.seed)
        for l in range(1, len(config.layer_sizes)):
            for no in range(0, len(self.ws[l-1])):
                for ni in range(0, len(self.ws[l-1][no])):
                    if self.config.random_weight:
                      w=(random.random()-0.5)
                    else:
                      w=((ni/len(self.ws[l-1][no]))-0.5)
                    self.ws[l-1][no][ni] = w*self.config.initial_weight_f
                if self.config.random_weight:
                    w=(random.random()-0.5)    
                else:
                    w=no / len(self.ws[l-1])  
                self.bs[l-1][no] = w*self.config.initial_weight_f
                    
        #print(self.ws)
        self.dh = 0
        self.dls = None
        self.adws = None
        self.dws = None
        self.adbs = None
        self.dbs = None
        self.reset_dls()
        self.reset_dws()

    # compute the  z+activation on lo
    # based on previous layer li and weight wo
    def compute_neuron(self, li, wo,b):
        z = 0
        ns = len(li)
        for ni in range(0, ns):
            z += li[ni] * wo[ni]
        z += b  # w[-1] is the bias

        wavg = z 
        if self.config.normalize_z:
            wavg/= len(wo)+1

        # activation function aka sigma
        a = self.config.sigma(wavg)
        return (wavg,a)

    # compute the layer lo
    def compute_layer(self, li, wio,bo, lo,zo):
        # print("compute_layer ",li)
        for no in range(0, len(lo)):
            # print(f"neuron {no}")
            z,a = self.compute_neuron(li, wio[no],bo[no])
            # print(f"neuron {no} ={v}")
            lo[no] = a
            zo[no] = z

    # forward computation
    def compute_network(self, inputs):
        # print("input",input)
        self.ls[0] = inputs
        for i in range(1, len(self.config.layer_sizes)):
            # print(f"layer {i}")
            self.compute_layer(self.ls[i - 1], self.ws[i - 1], self.bs[i-1],self.ls[i],self.zs[i])
            #print("layer ",i,":",self.ls[i])
        return self.ls[-1]


    # TODO put me in common code as compute_error(outputs, expecteds):
    def compute_error(self, inputs, expecteds):
        self.compute_network(inputs)
        e = error_function_avg(self.ls[-1], expecteds)
        
        # compute the correctness, valid only when output is a single category
        predicted=argmax(self.ls[-1])
        expected=argmax(expecteds)
        correct=1 if predicted==expected else 0
        return e,correct

    
    #
    # compute the cost over many samples (i.e. avg error on all samples)
    #
    #def cost(self, samples):
    ##    acc = 0
     #   results=[]
     #   for row in samples:
     #       # print(i)
     #       acc += self.compute_error(row[0], row[1])[0]
     #       results.append(self.ls[-1][:])
     #   e = acc / len(samples)
     #   return (e,results)

    def reset_dls(self):

        # dls is the derivatives of cost over neurons, needs to be reset after each sample
        self.dls = []
        self.dls.append(None) # we are not interested in the derivative of the cost over input layer
        for i in range(1, len(self.config.layer_sizes)):
            self.dls.append(zeros(self.config.layer_sizes[i]))

    def reset_dws(self):
        # derivative of cost over weights
        # accumulated derivative
        # note that the derivative of the error shows the opposite direction of the gradient we want to follow
        self.dws = []
        for li in range(1, len(self.config.layer_sizes)):
            self.dws.append(zeros2d(self.config.layer_sizes[li], self.config.layer_sizes[li - 1] ))

        self.dbs = []
        for li in range(1, len(self.config.layer_sizes)):
            self.dbs.append(zeros(self.config.layer_sizes[li]))
        # dh is the number of samples
        self.dh = 0

    def update_backtrack(self, inputs, expecteds):
        # L is last layer
        L = len(self.ls) - 1
        self.compute_network(inputs)
        e = error_function_avg(self.ls[-1], expecteds)
        # print("e",e)

        self.reset_dls()
        self.dh += 1
        # partial derivative of C over aLj (aLj: activation of last layer 's neuron j)
        for j in range(0, len(self.ls[L])):
            d_cost_aLj = error_functiond(self.ls[L][j], expecteds[j])
            self.dls[L][j] += d_cost_aLj

        # partial derivative of C over w[L-1]jk and b[L-1]j

        for l in range(L, 0, -1):
            for j in range(0, len(self.ls[l])):
                d_a_z = self.config.sigmad(self.zs[l][j])
                d_cost_a = self.dls[l][j]
                # weights
                for k in range(0, len(self.ls[l - 1])):
                    d_z_w = self.ls[l - 1][k]
                    if self.config.normalize_z:
                        d_z_w /= len(self.ls[l - 1])
                    d_cost_w = d_cost_a * d_a_z * d_z_w
                    self.dws[l - 1][j][k] += d_cost_w
                # bias
                d_z_b=1
                d_cost_b = d_cost_a * d_a_z * d_z_b
                self.dbs[l - 1][j] += d_cost_b

                if l > 1:
                # partial derivative of C over neurons for previous layer
                    for k in range(0, len(self.ls[l - 1])):
                        d_z_preva = self.ws[l - 1][j][k]                        
                        d_cost_preva=(d_z_preva*d_a_z*d_cost_a) #/ len(self.ls[l])
                        if self.config.normalize_z:
                            d_cost_preva /= len(self.ls[l])
                        self.dls[l - 1][k] += d_cost_preva

    def apply_dws(self):
        for l in range(1, len(self.config.layer_sizes)):
            for j in range(0, len(self.ls[l])):
                for k in range(0, len(self.ls[l-1])):  # also bias
                    gradientw=self.config.rate * self.dws[l - 1][j][k] / self.dh
                    self.ws[l - 1][j][k] -= gradientw
                gradientb=self.config.rate * self.dbs[l - 1][j] / self.dh
                self.bs[l - 1][j] -= gradientb
    
