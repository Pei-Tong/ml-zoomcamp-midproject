import pickle
import pandas as pd

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

if __name__ == "__main__":

    model_path = "./random_forest_model.pkl"
    model, dv = load_model(model_path)
    selected_features = ['age', 'annualincome', 'discountsavailed', 'loyaltyprogram', 'numberofpurchases', 'timespentonwebsite']

    customer = {
        'age': 47,
        'annualincome': 125446,
        'discountsavailed': 5,
        'loyaltyprogram': 'no',
        'numberofpurchases': 4,
        'timespentonwebsite': 16
    }
    
    purchase_probability = predict_customer(customer, model, dv, selected_features)
    print(f"Predicted Purchase probability: {purchase_probability:.3f}")


