# svm_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset path - kendi yolunu buraya yaz
file_path = "C:/Users/USER/telecom-customer-churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Veri yükleme
df = pd.read_csv(file_path)

# Basit ön işleme
df.drop(['customerID'], axis=1, inplace=True)  # customerID sütununu kaldır

# Kategorik değişkenleri encode et
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Hedef değişkeni encode et
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Eksik değer kontrolü ve kaldırma (varsa)
df.dropna(inplace=True)

# Özellikler ve hedef
X = df.drop('Churn', axis=1)
y = df['Churn']

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Özellikleri ölçeklendir (SVM için önemli)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model oluştur ve eğit
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Tahmin yap
y_pred = svm_model.predict(X_test)

# Performans metrikleri
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
