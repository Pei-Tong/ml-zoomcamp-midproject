import pickle
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        model, dv = pickle.load(f_in)
    return model, dv

def preprocess_input(customer, dv, selected_features):
    df = pd.DataFrame([customer]) 
    customer_dict = df.to_dict(orient='records') 
    X = dv.transform(customer_dict) 
    feature_names = dv.get_feature_names_out()
    selected_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
    X_selected = X[:, selected_indices]
    return X_selected

def predict_customer(customer, model, dv, selected_features):
    X_customer = preprocess_input(customer, dv, selected_features)
    y_pred = model.predict_proba(X_customer)[:, 1]
    return y_pred[0]

def main():
    st.title("Customer Purchase Prediction") 

    model_path = "./random_forest_model.pkl"
    model, dv = load_model(model_path)
    selected_features = ['age', 'annualincome', 'discountsavailed', 'loyaltyprogram', 'numberofpurchases', 'timespentonwebsite']

    with st.form("customer_form"):

        age = st.number_input("Age (18-70)", min_value=18, max_value=70, value=30, step=1)
        annual_income = st.slider("Annual Income (20,000-145,000)", min_value=20000, max_value=145000, value=50000, step=100)
        discounts_availed = st.slider("Discounts Availed", 0, 5, 2)
        loyalty_program = st.selectbox("Loyalty Program", ["yes", "no"])
        number_of_purchases = st.number_input("Number of Purchases (0-20)", min_value=0, max_value=20, value=10, step=1)
        time_spent_on_website = st.number_input("Stay Minutes (1-60)", min_value=1, max_value=60, value=20, step=1)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        customer = {
            'age': age,
            'annualincome': annual_income,
            'discountsavailed': discounts_availed,
            'loyaltyprogram': loyalty_program,
            'numberofpurchases': number_of_purchases,
            'timespentonwebsite': time_spent_on_website
        }

        purchase_probability = predict_customer(customer, model, dv, selected_features)
        
        st.success(f"Predicted Purchase Probability: {purchase_probability:.3f}")

if __name__ == "__main__":
    main()
