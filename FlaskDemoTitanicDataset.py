# Importing Required Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier



# 1. Data Collection from the source using Pandas

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Preview the dataset
print("Dataset Preview:")

# step 3: Handle missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# step 4: Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['Fare_log'] = np.log(df['Fare'] + 1.0)

# step 5: Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare_log', 'Embarked', 'FamilySize', 'IsAlone']
target = 'Survived'

X = df[features]
y = df[target]

# Step 6: Preprocessing pipeline
numeric_features = ['Age', 'Fare_log', 'FamilySize', 'IsAlone']
numeric_transformer = StandardScaler()

categorical_features = ['Sex', 'Embarked']
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 7: Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 8: Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Step 9: Save the model
joblib.dump(pipeline, 'titanic_model.pkl')