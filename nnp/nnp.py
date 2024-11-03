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
#
# todos:
# * organize code in classes
#
# * try on 1 dimensional function, check that it can learn any
# * try to make it a tiny bit faster (compile?)
# * try on handwritting samples
#
# * switch to a proper library, in order to move on
#


import pandas as pd
#import numpy as np
import math
import random
import matplotlib.pyplot as plt

random.seed(0)

# config: number of neurons in each layer
layersizes = [4, 12, 3]


# activation function
def sigma(z):
    v = math.tanh(z)
    # v=min(max(z,-1),1)
    return v


# derivative of activation function
# https://en.wikipedia.org/wiki/Activation_function
def sigmad(z):
    s = sigma(z)
    v = 1 - s * s
    return v


def zeros(s):
  v=[]
  for i in range(0,s):
    v.append(0)
  return v


def zeros2d(s1,s2):
  v=[]
  for i in range(0,s1):
    v.append(zeros(s2))
  return v



# layers
ls = []
for i in range(0, len(layersizes)):
    ls.append(zeros(layersizes[i]))

# weights
# ws[0] are the weights to update l[1] from l[0]
# ws[0][-1] is an extra weight which is the bias
ws = []
for li in range(1, len(layersizes)):
    ws.append(zeros2d(layersizes[li], layersizes[li - 1] + 1))

print(ws[0])
for wi in range(0, len(ws)):
    for no in range(0, len(ws[wi])):
        for ni in range(0, len(ws[wi][no])):
            ws[wi][no][ni] = random.random() * 2 - 1


# compute the activation on lo
# based on previous layer li and weight wo
def compute_neuron(li, wo, lo):
    z = 0
    ns = len(li)
    for ni in range(0, ns):
        z += li[ni] * wo[ni]
    z += wo[ns]  # w[-1] is the bias

    wavg = z / len(wo)

    # activation function aka sigma
    v = sigma(wavg)
    return v


# compute the layer lo
def compute_layer(li, wio, lo):
    # print("compute_layer ",li)
    for no in range(0, len(lo)):
        # print(f"neuron {no}")
        v = compute_neuron(li, wio[no], lo)
        # print(f"neuron {no} ={v}")
        lo[no] = v


# forward
def compute_network(input):
    # print("input",input)
    ls[0] = input
    for i in range(1, len(layersizes)):
        # print(f"layer {i}")
        compute_layer(ls[i - 1], ws[i - 1], ls[i])
    ls[-1]


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


def compute_error(input, expecteds):
    compute_network(input)
    e = error_function(ls[-1], expecteds)
    return e


#
# compute the cost (i.e. avg error on all samples)
#
def cost(df, idexes, input_cols, output_cols):

    acc = 0
    if idexes == None:
        idexes = range(0, len(df))
    for i in idexes:
        # print(i)
        acc += compute_error(
            list(df.iloc[i, input_cols]), list(df.iloc[i, output_cols])
        )

    e = acc / len(df)
    return e


# -- data --
def normalize_column(df, column):
    df[column] = df[column].apply(
        lambda x: (x - df[column].min()) / (df[column].max() - df[column].min())
    )


# read csv file
df = pd.read_csv("./data/iris.data", header=None)

input_cols = [0, 1, 2, 3]
output_cols = [5, 6, 7]


print(ws)
# normalize quantities
for column in input_cols:
    normalize_column(df, column)

# transform category into 0,1 columns
# there might be a simpler,native way to do that in panda
for category in list(pd.Categorical(df[4]).categories):
    df["is_" + category] = df[4].map(lambda x: 1 if x == category else 0)

# -- ride on --
# cost(df,[0,1,2,3],[5,6,7])


# to plot the activation
# d=list(np.arange(-2,2,0.1))
# v=list(map(lambda n: sigmad(n),d))
# plt.plot(d,v)
# plt.ylabel('sigmad')
# plt.show()


# backtracking

# L is last layer
L = len(ls) - 1

# note that the derivative of the error shows the oposite direction of the gradient we want to follow
# derivatives of cost over neurons
dls = []
for i in range(0, len(layersizes)):
    dls.append(zeros(layersizes[i]))

# derivative of cost over weights
dws = []
for li in range(1, len(layersizes)):
    dws.append(zeros2d(layersizes[li], layersizes[li - 1] + 1))

# accumulated derivative after each sample
adws = []
for li in range(1, len(layersizes)):
    adws.append(zeros2d(layersizes[li], layersizes[li - 1] + 1))

# dh is the number of samples
dh = 0


def update_backtrack(input, expecteds):

    compute_network(input)
    e = error_function(ls[-1], expecteds)
    # print("e",e)

    global dh
    dh = dh + 1
    # partial derivative of C over aLj (aLj: activation of last layer 's neuron j)
    for j in range(0, len(ls[L])):
        d_cost_aLj = 2 * (ls[L][j] - expecteds[j])
        dls[L][j] += d_cost_aLj

    # partial derivative of C over wLjk

    for l in range(L, 0, -1):
        for j in range(0, len(ls[l])):
            d_aj_z = sigmad(ls[l][j])
            for k in range(0, len(ls[l - 1])):
                d_z_w = ls[l - 1][k]
                d_cost_w = dls[l][j] * d_aj_z * d_z_w
                dws[l - 1][j][k] += d_cost_w
            # bias
            d_cost_w = dls[l][j] * d_aj_z * 1  # 1 for bias
            dws[l - 1][j][-1] += d_cost_w

        if l > 0:
            # partial derivative of C over neurons for previous layer
            for k in range(0, len(ls[l - 1])):
                for j in range(0, len(ls[l])):
                    dprev = dls[l][j]
                    dw = ws[l - 1][j][k]
                    da = sigmad(ls[l - 1][k])
                    dls[l - 1][k] += (dprev * dw * da) / len(ls[l])


def reset_derivatives():
    # print ("dh:",dh)
    for li in range(1, len(layersizes)):
        for ni in range(0, len(ls[li])):
            dls[li][ni] = 0

    for wi in range(0, len(ws)):
        for no in range(0, len(ws[wi])):
            for ni in range(0, len(ws[wi][no])):
                dws[wi][no][ni] = 0


def reset_acc_derivatives():
    global dh
    dh = 0
    for wi in range(0, len(ws)):
        for no in range(0, len(ws[wi])):
            for ni in range(0, len(ws[wi][no])):
                adws[wi][no][ni] = 0


def apply_acc_derivatives():
    for l in range(1, len(layersizes)):
        for j in range(0, len(ls[l])):
            for k in range(0, len(adws[l - 1][j])):  # also bias
                ws[l - 1][j][k] -= 0.3 * adws[l - 1][j][k] / dh


def acc_derivatives():
    for l in range(1, len(layersizes)):
        for j in range(0, len(ls[l])):
            for k in range(0, len(dws[l - 1][j])):  # also bias
                adws[l - 1][j][k] += dws[l - 1][j][k]


# preload weights from a previous session
wsXXX = [
            [2.8508284, 4.35413245, 1.51582444, 1.41718171, 6.66555368],
            [0.40836173, 3.90101729, 0.73337858, 2.46991725, 3.92161551],
            [1.06035039, 5.23707322, -0.3311598, 2.7290285, 3.84256655],
            [2.12286608, 2.91557033, 4.28594118, 3.90661503, 7.36541147],
            [-2.39806532, -5.2034498, 9.67634607, 12.05700448, -12.07488642],
            [3.22397035, -4.95446331, 8.2895549, 8.37162205, -1.90208526],
            [1.77076247, 3.24099072, 0.81269988, 1.41531344, 6.50820264],
            [-0.49843999, -3.77290689, 4.05880592, 5.25108488, 0.83214152],
            [0.60238345, -4.6223473, 7.51244755, 7.2590523, -0.43782064],
            [5.89228978, -5.74234178, 10.87857293, 11.05346111, -3.88841767],
            [-3.49149351, -9.64028735, 21.44740345, 23.36549073, -23.64950108],
            [7.96920738, -6.90856132, 12.7908494, 12.88749598, -5.38595298],
        ] , [
            [
                4.40027903,
                2.85790664,
                4.2294533,
                2.24273674,
                -1.39465099,
                -3.20286034,
                3.13303933,
                -1.46478743,
                -3.28645766,
                -6.39066321,
                2.03399102,
                -9.43940502,
                6.018206,
            ],
            [
                -0.97756044,
                -1.48103155,
                -2.29603587,
                0.30250333,
                0.17296399,
                2.18693903,
                0.11294457,
                -1.01422271,
                -0.66275886,
                4.62612947,
                -14.46599051,
                9.05497131,
                1.21887711,
            ],
            [
                2.5032802,
                2.23522262,
                2.54154719,
                2.92940963,
                6.12239159,
                0.23009993,
                1.51265725,
                2.57558068,
                2.44054551,
                -1.03023707,
                10.64333921,
                -5.27832321,
                2.28900566,
            ],
        ]



all = list(range(0, len(df)))
random.shuffle(all)
splitAt = int(len(df) * 0.9)
train = all[:splitAt]
test = all[splitAt:]
print("train", train)
print("test", test)
cost(df, test, input_cols, output_cols)

for x in range(1, 2000):
    sub_trains = train#np.split(np.array(train), 2)
    #for sub_train in sub_trains:
    for i in sub_trains:
            reset_derivatives()
            update_backtrack(
                list(df.iloc[i, input_cols]), list(df.iloc[i, output_cols])
            )
            # print("dh:",dh)
            acc_derivatives()

            # print("dws:",adws)
            # print("...")
    apply_acc_derivatives()
    reset_acc_derivatives()
    e=cost(df, test, input_cols, output_cols)
    print("x:",x," cost:", e)
    
    
    

print("ws", ws)
