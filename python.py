import os
print(f"Files will be saved in: {os.getcwd()}")
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv('breast_cancer_data.csv')

# 2. Preprocessing
# We drop 'id' and 'Unnamed: 32' as they aren't features
X = df.drop(['diagnosis', 'id'], axis=1, errors='ignore')
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
y = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. Scaling & Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. JOBLIB EXPORT 
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler exported successfully!")
