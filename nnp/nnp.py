import pandas as pd
import numpy as np
import math

np.random.seed(0)
# number of neurons in each layer
layersizes=[4,12,3]

# layers
ls=[]
for i in range(0,len(layersizes)):
  ls.append(np.zeros(layersizes[i]))

# weights
# ws[0] are the weights to update l[1] from l[0]
# ws[0][-1] is an extra weight which is the bias
ws=[]
for i in range(1,len(layersizes)):
    ws.append(np.random.random((layersizes[i]+1,layersizes[i-1])))


# compute the neuron no in layer lo
# based on previous layer li
def compute_neuron(li,w,lo):
    acc=0
    for ni in range(0,len(li)):
      acc+=li[ni]*w[ni]
    acc+=w[-1] # w[-1] is the bias
      

    # what the name used for the aggregator function in neural nets ?
    v=acc/len(li) # probably not this one...
    v=min(max(v,0),1)
    return v

   
# compute the layer lo
def compute_layer(li,wi,lo):
  for no in range(0,len(lo)):
    print(f"neuron {no}")
    v=compute_neuron(li,wi[no],lo)
    print(f"neuron {no} ={v}")
    lo[no]=v

# forward
def compute_network(input):
  ls[0]=input
  for i in range(1,len(layersizes)):
    print(f"layer {i}")
    compute_layer(ls[i-1],ws[i-1],ls[i])  
  ls[-1]  


# read csv file
df = pd.read_csv("./data/iris.data",header=None)


compute_network(list(df.iloc[0,0:3]))

