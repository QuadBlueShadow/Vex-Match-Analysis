import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("MatchData/PRACTICE 1")
#images = np.load("TrainingData/Qual32I.npy", allow_pickle=True)

print(df.head())
# df.plot(y='x')
# df.plot(y='y')
plt.scatter(x=df['x'], y=-df['y'])

plt.show()

# print(df.head())
# print(images[0])