import streamlit as st
import joblib
import numpy as np

# Load the saved model
with open('models\simple_linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)

# App Title
st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("💼 Salary Prediction App")
st.subheader("Predict your salary based on experience 📈")

# Sidebar for inputs
st.sidebar.header("Enter your details 👇")
experience = st.sidebar.slider("Years of Experience", min_value=0.0, max_value=20.0, step=0.5)

# Button to Predict
if st.sidebar.button("Predict Salary"):
    # Predict salary
    salary = model.predict(np.array([[experience]]))[0]

    # Display result
    st.success(f"🧾 Predicted Salary: ₹{salary:,.2f}")

    # Additional info
    st.info("💡 This prediction is based on a simple linear regression model.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")