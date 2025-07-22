import pandas as pd
import matplotlib.pyplot as plt

# Modellerin performans metrikleri
data = {
    'Model': [
        'Logistic Regression',
        'SVM',
        'Random Forest',
        'Neural Network',
        'Naive Bayes',
        'KNN',
        'Decision Tree'
    ],
    'Accuracy': [0.74, 0.74, 0.75, 0.75, 0.75, 0.73, 0.73],
    'Precision (1)': [0.53, 0.52, 0.55, 0.55, 0.52, 0.49, 0.50],
    'Recall (1)': [0.48, 0.48, 0.45, 0.45, 0.74, 0.46, 0.49],
    'F1-Score (1)': [0.50, 0.50, 0.49, 0.49, 0.61, 0.47, 0.50]
}

# DataFrame oluştur
df = pd.DataFrame(data)

# Tabloyu yazdır
print(df)

# Grafik çizimi
plt.figure(figsize=(12, 6))
bar_width = 0.15
index = range(len(df))

plt.bar([i - 1.5*bar_width for i in index], df['Accuracy'], width=bar_width, label='Accuracy')
plt.bar([i - 0.5*bar_width for i in index], df['Precision (1)'], width=bar_width, label='Precision (1)')
plt.bar([i + 0.5*bar_width for i in index], df['Recall (1)'], width=bar_width, label='Recall (1)')
plt.bar([i + 1.5*bar_width for i in index], df['F1-Score (1)'], width=bar_width, label='F1-Score (1)')

plt.xticks(index, df['Model'], rotation=30, ha='right')
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
