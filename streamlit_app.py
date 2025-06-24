import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('lasso_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🍺 Alcohol Consumption Prediction App")

# User inputs
country = st.text_input("Country")
continent = st.text_input("Continent")
beer_servings = st.number_input("Beer Servings", min_value=0)
spirit_servings = st.number_input("Spirit Servings", min_value=0)
wine_servings = st.number_input("Wine Servings", min_value=0)

if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'country': country,
        'beer_servings': beer_servings,
        'spirit_servings': spirit_servings,
        'wine_servings': wine_servings,
        'continent': continent
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Total Litres of Pure Alcohol: {round(prediction, 2)}")


    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv("C:\\Users\\ranji\\OneDrive\\Desktop\\Beer S\\beer-servings (1).csv")
df = df.dropna()

# Split features and target
X = df[['country', 'beer_servings', 'spirit_servings', 'wine_servings', 'continent']]
y = df['total_litres_of_pure_alcohol']

# One-hot encoder and pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'continent'])
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])

# Train-test split and fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model with your current sklearn version
with open("C:\\Users\\ranji\\OneDrive\\Desktop\\Beer App\\lasso_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved using scikit-learn 1.7.0")

