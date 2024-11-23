#
# NeuralNetConfig is used to define the neural network accross all samples
#
import pickle
import math
import random
import copy



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
        return None

    # derivative of activation function
    # https://en.wikipedia.org/wiki/Activation_function
    def sigmad(self, z: float) -> float:
        return None
        
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

RELUPYTORCH=NeuralNetConfig()
RELUPYTORCH.sigma=lambda x:0 if (x<0) else x
RELUPYTORCH.sigmad=lambda x:0 if (x<0) else 1


# error functions
# since these are not expected to change it is not in config

# single output neuron error function
def error_function(output1, expected1):
   return (output1 - expected1) * (output1 - expected1)

# single output neuron error function derivative
def error_functiond(output1, expected1):
   return 2*(output1 - expected1)

# output layer error
def error_function_avg(outputs, expecteds):
    acc2 = 0
    for i in range(0, len(expecteds)):
        acc2 += error_function(outputs[i], expecteds[i])
    
    e = acc2/ len(expecteds) 
    #print("outputs",outputs," expecteds",expecteds, " error:",e)
    return e



def argmax(values):
  return max(range(len(values)), key=values.__getitem__)


# compute the error + accuracy for a single run
def compute_metrics(outputs, expecteds):
        e = error_function_avg(outputs, expecteds)
        
        # compute the correctness, valid only when output is a single category
        predicted=argmax(outputs)
        expected=argmax(expecteds)
        correct=1 if predicted==expected else 0
        return e,correct

    
 # #
    # compute the cost over many samples (i.e. avg error on all samples)
    #
def cost(nn, samples,keep_output=False):
    e = 0
    a=0
    correct=0
    results=[]
    for row in samples:
        # print(i)
        outputs=nn.compute_network(row[0])
        e1,a1=compute_metrics(outputs,row[1])
    

        e+=e1
        a+=a1
        if (keep_output):
            results.append(outputs[:])
    e/= len(samples)
    a/= len(samples)
    return (e,a,results)



def learn(nn,data,realtest,epochs=1,iterations=10,use=1,rate_decay=0.8):
  #random.seed(0)
  for epoch in range(0, epochs):
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
    runs = cost(nn,realtest)
    print("epoch init cost:", runs[0],runs[2]," iterations:",iterations)
    for x in range(0, iterations):
        # for sub_train in sub_trains:
        for row in train:
            nn.update_backtrack(row[0],row[1])
        #print ("ws",nn.ws)
        #print ("dh",nn.dh,"dws:",nn.dws)
        print("iteration done")
        nn.apply_dws()
        nn.reset_dws()
        runs = cost(nn,realtest)
        print(f"epoch {epoch}/{epochs} iteration:{x}/{iterations} cost:", runs[0],runs[2])

    with open(f'tmp/weights-{epoch}.pickle', 'wb') as f:
      pickle.dump(nn.ws, f,protocol=pickle.HIGHEST_PROTOCOL)

    nn.config.rate*=rate_decay
    print(f"rate changed to {nn.config.rate}")    
    

    #print("ws", nn.ws)