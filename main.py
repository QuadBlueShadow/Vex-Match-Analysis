import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("MatchData/PRACTICE 1")
df2 = pd.read_csv("MatchData/PRACTICE 2")

#images = np.load("TrainingData/Qual32I.npy", allow_pickle=True)

print(df.head())
# df.plot(y='x')
# df.plot(y='y')
fig, ax = plt.subplots(2, 1)
ax[0].scatter(x=df['x4'], y=-df['y4'])

df = pd.concat([df, df2])

ax[1].scatter(x=df['x4'], y=-df['y4'])

plt.show()

# print(df.head())
# print(images[0])