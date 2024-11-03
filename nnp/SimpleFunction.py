
# test against a simple function
import random
from NeuralNet import *
import matplotlib.pyplot as plt


def f(x:float)->float:
  # if x>0:
  #    return 2-x
  return x*x


config=RELU.clone()
config.seed=12

config.layer_sizes=[1,10,1]
config.rate=0.1
nn=NeuralNet(config)

#nn.ws=[[[0.8,0.3]]]

precision=100
inputs=list(map(lambda x:x/precision,list(range(-2*precision,2*precision))))
outputs=list(map(lambda x:f(x),inputs))
data=list(map(lambda x:([x],[f(x)]),inputs))

def plotResults():
  e, os = nn.cost(data)
  plt.plot(inputs, outputs, inputs, os)
  plt.ylabel('in')
  plt.show()

plotResults()
for i in range (0,5):
  learn(nn,data,200)
  print("weights:",nn.ws)
  plotResults()




