import streamlit as st
import pickle
import pandas as pd

# Load files
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Customer Segmentation")

# Inputs
total_spend = st.number_input("Total Spend", 0.0)
total_orders = st.number_input("Total Orders", 0)
total_quantity = st.number_input("Total Quantity", 0)
unique_products = st.number_input("Unique Products", 0)

# Predict
if st.button("Predict"):

    input_df = pd.DataFrame([[
        total_spend,
        total_orders,
        total_quantity,
        unique_products
    ]], columns=features)

    scaled = scaler.transform(input_df)
    pca_data = pca.transform(scaled)
    cluster = model.predict(pca_data)[0]

    st.success(f"Customer belongs to Cluster: {cluster}")