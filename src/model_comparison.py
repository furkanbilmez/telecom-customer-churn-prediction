import matplotlib.pyplot as plt
import numpy as np
import os

# Define model names and metrics
models = ['Random Forest', 'SVM', 'Neural Net', 'Naive Bayes', 'Decision Tree', 'KNN']
accuracies = [0.74, 0.75, 0.75, 0.75, 0.73, 0.72]
f1_scores = [0.75, 0.75, 0.75, 0.76, 0.73, 0.73]

x = np.arange(len(models))
bar_width = 0.35

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the bars
bars_acc = ax.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='cornflowerblue')
bars_f1 = ax.bar(x + bar_width/2, f1_scores, bar_width, label='F1-Score', color='lightcoral')

# Customize the chart
ax.set_ylabel('Score')
ax.set_title('Comparison of ML Models Based on Accuracy and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.6, 0.85)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels on top of the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.01),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

add_value_labels(bars_acc)
add_value_labels(bars_f1)

# Create folder and save image
os.makedirs("model_comparison/images", exist_ok=True)
plt.tight_layout()
plt.savefig("model_comparison/images/model_comparison_chart.png")
plt.close()
