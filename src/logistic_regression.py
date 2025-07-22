# src/xgboost.py

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance
import os
 4 import pandas as pd
      5 import matplotlib.pyplot as plt
      6 import seaborn as sns
# Load dataset
data_path = 'C:/Users/USER/telecom-customer-churn-prediction/dataset/churn_clean.csv'
df = pd.read_csv(data_path)

# Drop unnecessary columns
df.drop(columns=['CustomerID', 'Count', 'Quarter', 'Referred a Friend', 'Number of Referrals', 'Avg Monthly GB Download'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and labels
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Create image directory if it doesn't exist
image_dir = 'C:/Users/USER/telecom-customer-churn-prediction/images'
os.makedirs(image_dir, exist_ok=True)

# Save Confusion Matrix as Image
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{image_dir}/xgboost_confusion_matrix.png')
plt.close()

# Save Feature Importance as Image
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10)
plt.title('Top 10 Feature Importances - XGBoost')
plt.tight_layout()
plt.savefig(f'{image_dir}/xgboost_feature_importance.png')
plt.close()
