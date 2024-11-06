#
# this file expects the mnist files in data folder
# the NN worked on first try (just adjusting the weights), so it is probably overfitting?
#

import numpy as np
import struct
import json
from NeuralNetNumpy import *

def get_data(marker):
  with open('./data/'+marker+'-images-idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    nrows, ncols = struct.unpack('>II', f.read(8))
    x_train = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")
    x_train = x_train.reshape((size,nrows*ncols))
    x_train = x_train/256 


  with open('./data/'+marker+'-labels-idx1-ubyte', 'rb') as i:
     magic, size = struct.unpack('>II', i.read(8))
     labels = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")    

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

data=data[:10000]
test=test[:1000]







config=RELU.clone()
config.seed=0

config.layer_sizes=[28*28,800,10]
config.rate=0.1
config.initial_weight_f=0.1
nn=NeuralNetNumpy(config)


for i in range(0,10):
  learn(nn,data,10,0.1)
  e = nn.cost(test)[0]
  print(f"after {i} trainings, test cost:", e)

  import pickle
  with open(f'mnist-weights-{i}.pickle', 'wb') as f:
    pickle.dump(nn.ws, f,protocol=pickle.HIGHEST_PROTOCOL)
