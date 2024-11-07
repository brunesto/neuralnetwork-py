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




# def learn(nn,data,iterations):
#   #random.seed(0)
#   data=data[:]
#   random.shuffle(data)

#   splitAt = int(len(data) * 1)
#   train = data[:splitAt]
#   test = data[splitAt:]
#   #print("train", train)
#   #print("test", test)

#   #print(" cost:", e)
#   e = nn.cost(data)[0]
#   print("itartion init cost:", e)
#   for x in range(1, iterations):
#       # for sub_train in sub_trains:
#       for row in train:
#           nn.update_backtrack(row[0],row[1])
#       #print ("ws",nn.ws)
#       #print ("dh",nn.dh,"dws:",nn.dws)

#       nn.apply_dws()
#       nn.reset_dws()
#       e = nn.cost(data)[0]
#       print("itartion:", x, " cost:", e)

#   #print("ws", nn.ws)


import pickle
def learn(nn,data,realtest,epochs=1,iterations=10,use=1,rate_decay=0.8):
  #random.seed(0)
  for epoch in range(1, epochs):
    data=data[:]
    random.shuffle(data)

    splitAt = int(len(data) * use)
    print(f"shuffling and cutting at {use:.0%} = index {splitAt}")
    train = data[:splitAt]
    test = data[splitAt:]
    #print("train", train)
    #print("test", test)

    #print(" cost:", e)
    print("data ready")
    e = nn.cost(realtest)[0]
    print("epoch init cost:", e)
    for x in range(1, iterations):
        # for sub_train in sub_trains:
        for row in train:
            nn.update_backtrack(row[0],row[1])
        #print ("ws",nn.ws)
        #print ("dh",nn.dh,"dws:",nn.dws)
        print("iteration done")
        nn.apply_dws()
        nn.reset_dws()
        e = nn.cost(realtest)[0]
        print(f"epoch {epoch}/{epochs} iteration:{x}/{iterations} cost:", e)

    with open(f'tmp/weights-{epoch}.pickle', 'wb') as f:
      pickle.dump(nn.ws, f,protocol=pickle.HIGHEST_PROTOCOL)

    nn.config.rate*=rate_decay
    print(f"rate changed to {nn.config.rate}")    
    

    #print("ws", nn.ws)