import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("C:\\Users\\ranji\\OneDrive\\Desktop\\Beer S\\beer-servings (1).csv")
df = df.dropna()

X = df[['country', 'beer_servings', 'spirit_servings', 'wine_servings', 'continent']]
y = df['total_litres_of_pure_alcohol']

# Preprocessing and pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'continent'])
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model using current scikit-learn version
with open("C:\\Users\\ranji\\OneDrive\\Desktop\\Beer App\\lasso_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully with your current scikit-learn version")
