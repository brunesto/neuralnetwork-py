import pandas as pd
import numpy as np
import math

np.random.seed(0)

# config: number of neurons in each layer
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
    v=acc/len(w) # probably not this one...
    v=min(max(v,0),1)
    return v

   
# compute the layer lo
def compute_layer(li,wi,lo):
  print("compute_layer ",li)
  for no in range(0,len(lo)):
    #print(f"neuron {no}")
    v=compute_neuron(li,wi[no],lo)
    #print(f"neuron {no} ={v}")
    lo[no]=v

# forward
def compute_network(input):
  print("input",input)
  ls[0]=input
  for i in range(1,len(layersizes)):
    #print(f"layer {i}")
    compute_layer(ls[i-1],ws[i-1],ls[i])  
  ls[-1]  


def error_function(outputs,expecteds):
   acc2=0
   for i in range(0,len(expecteds)):
      output=outputs[i]
      expected=expecteds[i]
      acc2+=(output-expected)*(output-expected)
   e=math.sqrt(acc2)   
   print("outputs",outputs," expecteds",expecteds, " error:",e)
   return e

def compute_error(input,expecteds):
   compute_network(input)
   e=error_function(ls[-1],expecteds)
   return e

#
# compute the cost (i.e. avg error on all samples)
#
def cost(df,input_cols,output_cols):

  acc=0
  for  i in range(0,len(df)):
    print(i)
    acc+=compute_error(list(df.iloc[i,input_cols]),list(df.iloc[i,output_cols]))

  e=acc/len(df)
  print("cost:",e)

# -- data --
def normalize_column(df,column):
  df[column] = df[column].apply(lambda x: (x - df[column].min()) / (df[column].max() - df[column].min()))

# read csv file
df = pd.read_csv("./data/iris.data",header=None)

# normalize quantities
for column in [0,1,2,3]:
  normalize_column(column)

# transform category into 0,1 columns
# there might be a simpler,native way to do that in panda
for category in list(pd.Categorical(df[4]).categories):
   df['is_'+category] = df[4].map(lambda x: 1 if x == category else 0)

# -- ride on --
cost(df,[0,1,2,3],[5,6,7])


