import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('movie_revenue_model.pkl', 'rb'))

st.title("🎬 Movie Revenue Prediction App")
st.write("This app predicts the estimated box office revenue of a movie based on its key features.")

# Input fields
budget = st.number_input("Enter Movie Budget (in USD $)", 
                         min_value=100000, max_value=500000000, step=100000)
popularity = st.slider("Popularity Index", 0.0, 200.0, 50.0)
runtime = st.slider("Runtime (in minutes)", 60, 240, 120)
vote_average = st.slider("Average Audience Rating (0–10)", 0.0, 10.0, 5.0)

# Predict button
if st.button("Predict Revenue"):
    # Prepare features for model
    features = np.array([[np.log1p(budget), popularity, runtime, vote_average]])
    
    # Predict log revenue and convert back to actual value
    predicted_log_revenue = model.predict(features)[0]
    predicted_revenue_usd = np.expm1(predicted_log_revenue)

    # Convert USD → INR (approximate rate)
    usd_to_inr = 85
    predicted_revenue_inr = predicted_revenue_usd * usd_to_inr
    
    # Convert INR → Crores for readability
    predicted_revenue_crore = predicted_revenue_inr / 1e7

    # Display results
    st.subheader("💰 Predicted Revenue")
    st.write(f"**In USD:** ${predicted_revenue_usd:,.2f}")
    st.write(f"**In INR:** ₹{predicted_revenue_inr:,.0f}")
    st.success(f"**Approx. ₹{predicted_revenue_crore:,.2f} crore** (Indian Box Office Estimate)")

    # Additional note
    st.caption("Note: Predictions are based on available data trends and should be interpreted as estimates.")
