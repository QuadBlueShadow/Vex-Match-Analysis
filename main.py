import pandas as pd
import numpy as np

df = pd.read_csv("TrainingData/Qual32")
images = np.load("TrainingData/Qual32I.npy", allow_pickle=True)

print(df.head())
print(images[0])