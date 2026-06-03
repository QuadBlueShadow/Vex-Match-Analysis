import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from database import DataBase

# db = DataBase("InnovateDivision", ["results", "match VARCHAR (25)", "Red_1 VARCHAR (10)", "Red_2 VARCHAR (10)", "Blue_1 VARCHAR (10)", "Blue_2 VARCHAR (10)", "Red_Score INTEGER", "Blue_Score INTEGER", "Winner VARCHAR (5)"])
# db.insert_data(l, ["match", "Red_1", "Red_2", "Blue_1", "Blue_2", "Red_Score", "Blue_Score", "Winner"], "results")

db = DataBase("InnovateDivision", ["PRACTICE 5".replace(" ", "_").lower(), "x1 FLOAT", "y1 FLOAT", "x2 FLOAT", "y2 FLOAT", "x3 FLOAT", "y3 FLOAT", "x4 FLOAT", "y4 FLOAT"])

table_parameters = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
db.insert_data([[0, 0, 0, 0, 0, 0, 0, 0]], table_parameters, "practice_5")
print(len(db.get_data("SELECT * FROM practice_5")))

db.close_connection()