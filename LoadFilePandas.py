import pandas as pd
import matplotlib# Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# reading the database
data = pd.read_csv("tips.csv")

plt.title("Scatter Plot")
plt.xlabel("Day")
plt.ylabel("Tip")

plt.scatter(data["day"], data["tip"])
