import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df['gender'] = df['gender'].map({0: 'male', 1: 'female'})
    df['productcategory'] = df['productcategory'].map({0: 'electronics', 1: 'clothing', 2: 'homegoods', 3: 'beauty', 4: 'sports'})
    df['loyaltyprogram'] = df['loyaltyprogram'].map({0: 'no', 1: 'yes'})
    df['purchasestatus'] = df['purchasestatus'].map({0: 'no', 1: 'yes'})
    return df

def preprocess_data(df, selected_features):
    y = (df['purchasestatus'] == 'yes').astype(int).values
    df = df.drop(columns=['purchasestatus'])
    train_dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(train_dicts)
    feature_names = dv.get_feature_names_out()
    selected_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
    X = X[:, selected_indices]
    return X, y, dv

def train_and_evaluate(df, selected_features):
    X, y, dv = preprocess_data(df, selected_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=110, max_depth=15, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    with open('random_forest_model.pkl', 'wb') as f_out:
        pickle.dump((rf, dv), f_out)
    print("Model and DictVectorizer saved!")
