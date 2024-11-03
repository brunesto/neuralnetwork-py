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
import copy


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
    rate = 0.1
    # seed for populating initial weights
    seed = 0

    normalize_z=False
    random_weight=True
    initial_weight_f=5

    # activation function
    def sigma(self, z: float) -> float:
        v = math.tanh(z)
        # v=min(max(z,-1),1)
        return v

    # derivative of activation function
    # https://en.wikipedia.org/wiki/Activation_function
    def sigmad(self, z: float) -> float:
        s = math.tanh(z)
        v = 1 - s * s
        return v
        
    def clone(self):
        return copy.deepcopy(self)

# it seems that
# TANH is not able to grow past the original weights
# 
TANH=NeuralNetConfig()
TANH.sigma=lambda x:math.tanh(x)
def tanhd(x):
    s = math.tanh(x)
    v = 1 - s * s
    return v

TANH.sigmad=tanhd
    

LINEAR=NeuralNetConfig()
LINEAR.sigma=lambda x:x
LINEAR.sigmad=lambda x:1

RELU=NeuralNetConfig()
RELU.sigma=lambda x:0.1*x if (x<0) else x
RELU.sigmad=lambda x:0.1 if (x<0) else 1
# error functions
# since these are not expected to change it is not in config

# single output neuron error function
def error_function(output1, expected1):
   return (output1 - expected1) * (output1 - expected1)

# single output neuron error function derivative
def error_functiond(output1, expected1):
   return 2*(output1 - expected1)

# output layer error
def error_function_acc(outputs, expecteds):
    acc2 = 0
    for i in range(0, len(expecteds)):
        acc2 += error_function(outputs[i], expecteds[i])
    
    e = acc2  
    # print("outputs",outputs," expecteds",expecteds, " error:",e)
    return e


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
            self.ws.append(zeros2d(config.layer_sizes[li], config.layer_sizes[li - 1] + 1))

        # now randomly populate weights
        random.seed(self.config.seed)
        for wi in range(0, len(self.ws)):
            for no in range(0, len(self.ws[wi])):
                for ni in range(0, len(self.ws[wi][no])):
                    if self.config.random_weight:
                      w=(random.random()-0.5)
                    else:
                      w=((ni/len(self.ws[wi][no]))-0.5)
                    self.ws[wi][no][ni] = w*self.config.initial_weight_f
        print(self.ws)
        self.dh = 0
        self.dls = None
        self.adws = None
        self.dws = None
        self.reset_dls()
        self.reset_dws()

    # compute the  z+activation on lo
    # based on previous layer li and weight wo
    def compute_neuron(self, li, wo):
        z = 0
        ns = len(li)
        for ni in range(0, ns):
            z += li[ni] * wo[ni]
        z += wo[ns]  # w[-1] is the bias

        wavg = z 
        if self.config.normalize_z:
            wavg/= len(wo)

        # activation function aka sigma
        a = self.config.sigma(wavg)
        return (wavg,a)

    # compute the layer lo
    def compute_layer(self, li, wio, lo,zo):
        # print("compute_layer ",li)
        for no in range(0, len(lo)):
            # print(f"neuron {no}")
            z,a = self.compute_neuron(li, wio[no])
            # print(f"neuron {no} ={v}")
            lo[no] = a
            zo[no] = z

    # forward computation
    def compute_network(self, inputs):
        # print("input",input)
        self.ls[0] = inputs
        for i in range(1, len(self.config.layer_sizes)):
            # print(f"layer {i}")
            self.compute_layer(self.ls[i - 1], self.ws[i - 1], self.ls[i],self.zs[i])
        return self.ls[-1]

    def compute_error(self, inputs, expecteds):
        self.compute_network(inputs)
        e = error_function_acc(self.ls[-1], expecteds)
        return e

    #
    # compute the cost over many samples (i.e. avg error on all samples)
    #
    def cost(self, samples):
        acc = 0
        results=[]
        for row in samples:
            # print(i)
            acc += self.compute_error(row[0], row[1])
            results.append(self.ls[-1][:])
        e = acc / len(samples)
        return (e,results)

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
            self.dws.append(zeros2d(self.config.layer_sizes[li], self.config.layer_sizes[li - 1] + 1))
        # dh is the number of samples
        self.dh = 0

    def update_backtrack(self, inputs, expecteds):
        # L is last layer
        L = len(self.ls) - 1
        self.compute_network(inputs)
        e = error_function_acc(self.ls[-1], expecteds)
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
                self.dws[l - 1][j][-1] += d_cost_b

                if l > 1:
                # partial derivative of C over neurons for previous layer
                #for j in range(0, len(self.ls[l])):
                #    d_a_z = self.config.sigmad(self.zs[l][j])
                #    d_cost_a = self.dls[l][j]
                    for k in range(0, len(self.ls[l - 1])):
                        d_z_preva = self.ws[l - 1][j][k]                        
                        d_cost_preva=(d_z_preva*d_a_z*d_cost_a) #/ len(self.ls[l])
                        if self.config.normalize_z:
                            d_cost_preva /= len(self.ls[l])
                        self.dls[l - 1][k] += d_cost_preva

    def apply_dws(self):
        for l in range(1, len(self.config.layer_sizes)):
            for j in range(0, len(self.ls[l])):
                for k in range(0, len(self.ls[l-1])+1):  # also bias
                    gradient=self.config.rate * self.dws[l - 1][j][k] / self.dh
                    #print(gradient)
                    #gradient=-math.sqrt(gradient) if (gradient>0) else math.sqrt(-gradient)
                    self.ws[l - 1][j][k] -= gradient




def learn(nn,data,iterations):
  #random.seed(0)
  data=data[:]
  random.shuffle(data)

  splitAt = int(len(data) * 1)
  train = data[:splitAt]
  test = data[splitAt:]
  #print("train", train)
  #print("test", test)

  #print(" cost:", e)
  e = nn.cost(data)[0]
  print("itartion init cost:", e)
  for x in range(1, iterations):
      # for sub_train in sub_trains:
      for row in train:
          nn.update_backtrack(row[0],row[1])
      #print ("ws",nn.ws)
      #print ("dh",nn.dh,"dws:",nn.dws)

      nn.apply_dws()
      nn.reset_dws()
      e = nn.cost(data)[0]
      print("itartion:", x, " cost:", e)

  #print("ws", nn.ws)