import numpy as np
import matplotlib.pyplot as plt
from validate_run import validate_run
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl

mpl.rc('font', family='Times New Roman', size=24)
df = pd.read_csv('/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/AC_score.csv')
df = df.set_index('Models')

# Generate some data
x = [14.3, 15.66]  # Example x data
y = [7.091649524, 7.327807291]  # Example y data

a = [0.4699853516, 0.5338916018]  # Example a data
b = [4.59365364, 7.537056766]  # Example b data

# Create the plot
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(x, y, color=(165/255, 165/255, 165/255), linestyle='-', marker='', markersize=36, linewidth=3)
# plt.plot(a, b, color=(165/255, 165/255, 165/255), linestyle='-', marker='', markersize=36, linewidth=3)

plt.plot(x[0], y[0], marker='o', markersize=10, label="A", color=(103/255,151/255,194/255))
plt.plot(x[1], y[1], marker='o', markersize=10, label="B", color=(242/255,164/255,92/255))

# Adjusting the tick frequency
# print(min(x))
# x_ticks = np.arange(min(a), max(a) + 0.01, 0.02)  # Adjust the step size for more ticks on the x-axis
# y_ticks = np.arange(min(b), max(b) + 0.1, 0.4)  # Adjust the step size for more ticks on the y-axis

x_ticks = np.arange(min(x), max(x) + 0.1, 0.2)  # Adjust the step size for more ticks on the x-axis
y_ticks = np.arange(min(y), max(y) + 0.1, 0.05)  # Adjust the step size for more ticks on the y-axis

plt.xticks(x_ticks)
plt.yticks(y_ticks)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Show the plot
plt.show()
