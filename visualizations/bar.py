import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

# Data
labels = ['Random', 'A Score', 'C Score', 'AC Score']
percentages = [36.90, 91.48, 56.56, 94.06]

# Create bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, percentages, alpha=0.75)

# Adding title and labels
# plt.xlabel('Factors')
plt.ylabel('Coefficient of Determination (%)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)


# Display the bar graph
plt.savefig('bar_chart_nature.png', dpi=300)
