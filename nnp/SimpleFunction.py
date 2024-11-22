
# test against a simple function
import random
from NeuralNet import *
from NeuralNetNumpy import *
import matplotlib.pyplot as plt

def f(x:float)->float:
  # if x>0:
  #    return 2-x
  return x*x*x


config=RELU.clone()
config.seed=0

config.layer_sizes=[1,5,5,1]
config.rate=0.1

nn=NeuralNet(config)

#nn.ws=[[[0.8,0.3]]]

precision=100
inputs=list(map(lambda x:[x/precision],list(range(-2*precision,2*precision))))
outputs=list(map(lambda x:[f(x[0])],inputs))
data=list(map(lambda x:(x,[f(x[0])]),inputs))


predicted=nn.compute_network(data[0][0])
compute_metrics(predicted, data[0][1])

def plotResults():
  e, accuracies,os = cost(nn,data,keep_output=True)
  plt.plot(inputs, outputs, inputs, os)
  plt.ylabel('in')
  plt.show()




plotResults()

  
learn(nn,data,data,5,30)
  #print("weights:",nn.ws)
plotResults()
#  config.rate*=0.9




