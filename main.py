import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from database import DataBase

df = pd.read_csv("MatchData/InnovateDivisionResults.csv")

#print(df[df.isna().any(axis=1)])

# l = df.iloc[:, 1:].to_numpy().tolist()
# print(l)

# db = DataBase("InnovateDivision", ["results", "match VARCHAR (25)", "Red_1 VARCHAR (10)", "Red_2 VARCHAR (10)", "Blue_1 VARCHAR (10)", "Blue_2 VARCHAR (10)", "Red_Score INTEGER", "Blue_Score INTEGER", "Winner VARCHAR (5)"])
# db.insert_data(l, ["match", "Red_1", "Red_2", "Blue_1", "Blue_2", "Red_Score", "Blue_Score", "Winner"], "results")