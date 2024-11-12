#
# this file expects the mnist files in data folder
# the NN worked on first try (just adjusting the weights), so it is probably overfitting?
#

import numpy as np
import struct
import json
import pickle
from NeuralNetNumpy import *
from NeuralNet import *


def get_data(marker):
  with open('./data/'+marker+'-images-idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    nrows, ncols = struct.unpack('>II', f.read(8))
    x_train = np.fromfile(f, dtype=np.dtype(np.uint8))#.newbyteorder(">")
    x_train = x_train.reshape((size,nrows*ncols))
    x_train = x_train/256 


  with open('./data/'+marker+'-labels-idx1-ubyte', 'rb') as i:
     magic, size = struct.unpack('>II', i.read(8))
     labels = np.fromfile(i, dtype=np.dtype(np.uint8))#.newbyteorder(">")    

  y_train=[]
  for label in labels:
   col=np.zeros(10)
   col[label]=1
   y_train.append(col)
  data=list(zip(x_train,y_train))     
  return data

data=get_data("train")
test=get_data("t10k")


print("read train  data:",len(data))
print("read test data:",len(test))

data=data
test=test[:1000]







config=RELU.clone()
config.seed=0

config.layer_sizes=[28*28,800,10]
config.rate=0.1
config.initial_weight_f=0.1
nn=NeuralNetNumpy(config)




#with open('./computed/mnist-weights-9.pickle', 'rb') as f:
#  nn.ws=pickle.load( f)


learn(nn,data,test,epochs=1,iterations=1,use=0.1,rate_decay=0.8)


def pretty_print_input(sample,res=None):
  idx=0
  for y in range(0,28):
    for x in range(0,28):
      print(" " if sample[idx]<0.1 else "x",end="")
      idx+=1
    print()  

def pretty_print_answer(res):
  if res is not None:
    answers=0
    for i in range(0,10):
      if (res[i]>0.2):
        print(i,res[i])
        answers+=1
    if answers==0:
        print("no candidate")

# dump the first 100 digits
for i in range(0,0):
  pretty_print_input(test[i][0])
  print ("expected:")
  pretty_print_answer(test[i][1])
  print ("output:")
  pretty_print_answer(nn.compute_network(test[i][0]))


#import code; 
#code.interact(local=locals())
