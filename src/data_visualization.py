import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset path (full path)
data_path = "C:/Users/USER/telecom-customer-churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load data
df = pd.read_csv(data_path)

# Basic preprocessing
# Drop customerID since it's an identifier
df = df.drop(columns=['customerID'])

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Split data into features and target
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
