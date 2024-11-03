import pandas as pd
from NeuralNet import *


# -- data --
def normalize_column(df, column):
    df[column] = df[column].apply(
        lambda x: (x - df[column].min()) / (df[column].max() - df[column].min())
    )


nn = NeuralNet(NeuralNetConfig())
# read csv file
df = pd.read_csv("../data/iris.data", header=None)

input_cols = [0, 1, 2, 3]
output_cols = [5, 6, 7]

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
], [
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
nn.cost(df, test, input_cols, output_cols)

for x in range(1, 100):
    sub_trains = train  # np.split(np.array(train), 2)
    nn.init_derivatives()
    # for sub_train in sub_trains:
    for i in sub_trains:
        nn.reset_derivatives()
        nn.update_backtrack(
            list(df.iloc[i, input_cols]), list(df.iloc[i, output_cols])
        )
        # print("dh:",dh)


        # print("dws:",adws)
        # print("...")
    nn.acc_derivatives()
    nn.apply_acc_derivatives()
    nn.reset_acc_derivatives()
    e = nn.cost(df, test, input_cols, output_cols)
    print("x:", x, " cost:", e)

print("ws", nn.ws)
