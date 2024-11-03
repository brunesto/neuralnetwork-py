#  resources
#  python:
#  https://github.com/blu3r4y/python-for-java-developers, couple of hours reading
#  https://gto76.github.io/python-cheatsheet
#
#  3Blue1Brown, series on neural networks  https://www.youtube.com/watch?v=Ilg3gGewQ5U
#
#  Also a bit of proper AI help comes in handy: there was a bug in the derivation of bias in the 1st version of backtracking, ChatGPT found it
#
# done:
# * forward cost
# * simple backtracking
# * classes
#
# todos:
# 
#
# * try on 1 dimensional function, check that it can learn any
# * try to make it a tiny bit faster (compile?)
# * try on handwritting samples
#
# * switch to a proper library, in order to move on
#


import math
import random


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


# configuration for neural network
class NeuralNetConfig:
    # config: number of neurons in each layer
    layer_sizes = [4, 12, 3]
    # learning rate
    rate = 0.3
    # seed for populating initial weights
    seed = 0

    # activation function
    def sigma(self, z: float) -> float:
        v = math.tanh(z)
        # v=min(max(z,-1),1)
        return v

    # derivative of activation function
    # https://en.wikipedia.org/wiki/Activation_function
    def sigmad(self, z: float) -> float:
        s = self.sigma(z)
        v = 1 - s * s
        return v


def error_function(outputs, expecteds):
    acc2 = 0
    for i in range(0, len(expecteds)):
        output = outputs[i]
        expected = expecteds[i]
        acc2 += (output - expected) * (output - expected)
    # question: is it necessary to sqrt?
    e = acc2  # math.sqrt(acc2)
    # print("outputs",outputs," expecteds",expecteds, " error:",e)
    return e


class NeuralNet:
    config: NeuralNetConfig
    # layers
    ls = []
    # weights
    # ws[0] are the weights to update l[1] from l[0]
    # ws[0][-1] is an extra weight which is the bias
    ws = []

    def __init__(self, config):
        self.config = config
        for i in range(0, len(config.layer_sizes)):
            self.ls.append(zeros(config.layer_sizes[i]))

        for li in range(1, len(config.layer_sizes)):
            self.ws.append(zeros2d(config.layer_sizes[li], config.layer_sizes[li - 1] + 1))

        # now randomly populate weights
        random.seed(self.config.seed)
        for wi in range(0, len(self.ws)):
            for no in range(0, len(self.ws[wi])):
                for ni in range(0, len(self.ws[wi][no])):
                    self.ws[wi][no][ni] = random.random() * 2 - 1

    # compute the activation on lo
    # based on previous layer li and weight wo
    def compute_neuron(self, li, wo, lo):
        z = 0
        ns = len(li)
        for ni in range(0, ns):
            z += li[ni] * wo[ni]
        z += wo[ns]  # w[-1] is the bias

        wavg = z / len(wo)

        # activation function aka sigma
        v = self.config.sigma(wavg)
        return v

    # compute the layer lo
    def compute_layer(self, li, wio, lo):
        # print("compute_layer ",li)
        for no in range(0, len(lo)):
            # print(f"neuron {no}")
            v = self.compute_neuron(li, wio[no], lo)
            # print(f"neuron {no} ={v}")
            lo[no] = v

    # forward computation
    def compute_network(self, inputs):
        # print("input",input)
        self.ls[0] = inputs
        for i in range(1, len(self.config.layer_sizes)):
            # print(f"layer {i}")
            self.compute_layer(self.ls[i - 1], self.ws[i - 1], self.ls[i])
        return self.ls[-1]

    def compute_error(self, inputs, expecteds):
        self.compute_network(inputs)
        e = error_function(self.ls[-1], expecteds)
        return e

    #
    # compute the cost (i.e. avg error on all samples)
    #
    def cost(self, df, idexes, input_cols, output_cols):

        acc = 0
        if idexes is None:
            idexes = range(0, len(df))
        for i in idexes:
            # print(i)
            acc += self.compute_error(
                list(df.iloc[i, input_cols]), list(df.iloc[i, output_cols])
            )

        e = acc / len(df)
        return e

    def init_derivatives(self):
        # note that the derivative of the error shows the oposite direction of the gradient we want to follow
        # derivatives of cost over neurons
        self.dls = []
        for i in range(0, len(self.config.layer_sizes)):
            self.dls.append(zeros(self.config.layer_sizes[i]))

        # derivative of cost over weights
        self.dws = []
        for li in range(1, len(self.config.layer_sizes)):
            self.dws.append(zeros2d(self.config.layer_sizes[li], self.config.layer_sizes[li - 1] + 1))

        # accumulated derivative after each sample
        self.adws = []
        for li in range(1, len(self.config.layer_sizes)):
            self.adws.append(zeros2d(self.config.layer_sizes[li], self.config.layer_sizes[li - 1] + 1))

        # dh is the number of samples
        self.dh = 0

    def update_backtrack(self, input, expecteds):
        # L is last layer
        L = len(self.ls) - 1
        self.compute_network(input)
        e = error_function(self.ls[-1], expecteds)
        # print("e",e)

        self.dh = self.dh + 1
        # partial derivative of C over aLj (aLj: activation of last layer 's neuron j)
        for j in range(0, len(self.ls[L])):
            d_cost_aLj = 2 * (self.ls[L][j] - expecteds[j])
            self.dls[L][j] += d_cost_aLj

        # partial derivative of C over wLjk

        for l in range(L, 0, -1):
            for j in range(0, len(self.ls[l])):
                d_aj_z = self.config.sigmad(self.ls[l][j])
                for k in range(0, len(self.ls[l - 1])):
                    d_z_w = self.ls[l - 1][k]
                    d_cost_w = self.dls[l][j] * d_aj_z * d_z_w
                    self.dws[l - 1][j][k] += d_cost_w
                # bias
                d_cost_w = self.dls[l][j] * d_aj_z * 1  # 1 for bias
                self.dws[l - 1][j][-1] += d_cost_w

            if l > 0:
                # partial derivative of C over neurons for previous layer
                for k in range(0, len(self.ls[l - 1])):
                    for j in range(0, len(self.ls[l])):
                        dprev = self.dls[l][j]
                        dw = self.ws[l - 1][j][k]
                        da = self.config.sigmad(self.ls[l - 1][k])
                        self.dls[l - 1][k] += (dprev * dw * da) / len(self.ls[l])

    def reset_derivatives(self):
        # print ("dh:",dh)
        for li in range(1, len(self.config.layer_sizes)):
            for ni in range(0, len(self.ls[li])):
                self.dls[li][ni] = 0

        for wi in range(0, len(self.ws)):
            for no in range(0, len(self.ws[wi])):
                for ni in range(0, len(self.ws[wi][no])):
                    self.dws[wi][no][ni] = 0

    def reset_acc_derivatives(self):
        global dh
        dh = 0
        for wi in range(0, len(self.ws)):
            for no in range(0, len(self.ws[wi])):
                for ni in range(0, len(self.ws[wi][no])):
                    self.adws[wi][no][ni] = 0

    def apply_acc_derivatives(self):
        for l in range(1, len(self.config.layer_sizes)):
            for j in range(0, len(self.ls[l])):
                for k in range(0, len(self.adws[l - 1][j])):  # also bias
                    self.ws[l - 1][j][k] -= self.config.rate * self.adws[l - 1][j][k] / self.dh

    def acc_derivatives(self):
        for l in range(1, len(self.config.layer_sizes)):
            for j in range(0, len(self.ls[l])):
                for k in range(0, len(self.dws[l - 1][j])):  # also bias
                    self.adws[l - 1][j][k] += self.dws[l - 1][j][k]
