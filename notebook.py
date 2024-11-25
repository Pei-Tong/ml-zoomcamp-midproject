# %% [markdown]
# # Predict Customer Purchase Behavior
# Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import pickle

%matplotlib inline

# %% [markdown]
# ### A. Data preparation and data cleaning
# * Downloading the dataset
# * Re-encoding the categorical variables
# * Doing the train/validation/test split

# %%
data_path = './customer_purchase_data.csv'
df = pd.read_csv(data_path)

# %%
data_info = df.info()
data_head = df.head()
data_missing = df.isnull().sum()
data_describe = df.describe()

# %%
df.max()

# %%
df.columns = df.columns.str.lower()

# %%
df.columns

# %%
df.head()

# %% [markdown]
# #### Distinguish between numerical features and categorical features

# %%
categorical_features = ['gender', 'productcategory', 'loyaltyprogram', 'purchasestatus']
numerical_features = [col for col in df.columns if col not in categorical_features]

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# %%
gender_values = {
    0: 'male',
    1: 'female'
}

df.gender = df.gender.map(gender_values)

productcategory_values = {
    0: 'electronics', 
    1: 'clothing',
    2: 'homegoods',
    3: "beauty",
    4: "sports"
}

df.productcategory = df.productcategory.map(productcategory_values)


loyaltyprogram_values = {
    0: 'no',
    1: 'yes'
}

df.loyaltyprogram = df.loyaltyprogram.map(loyaltyprogram_values)


purchasestatus_values = {
    0: 'no',
    1: 'yes'
}

df.purchasestatus = df.purchasestatus.map(purchasestatus_values)

# %% [markdown]
# ### B. EDA, feature importance analysis

# %%
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()

# %% [markdown]
# #### Conduct preliminary analysis of numerical features

# %%
# Draw a histogram and add the kernel density estimation curve (KDE Curve)
for col in numerical_features:
    sns.histplot(df[col], kde=True) # Draw a histogram, use KDE to smooth data distribution
    plt.title(col)
    plt.show()


# %%
for col in categorical_features:
    print(f'Column: {col}, Unique values: {df[col].nunique()}')
    sns.countplot(x=col, data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# #### Setting up the Validation Framework

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
y_train = (df_train.purchasestatus == 'yes').astype('int').values
y_val = (df_val.purchasestatus == 'yes').astype('int').values
y_test = (df_test.purchasestatus == 'yes').astype('int').values

# %%
del df_train['purchasestatus']
del df_val['purchasestatus']
del df_test['purchasestatus']

# %%
df_train

# %% [markdown]
# #### feature importance analysis

# %%
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

# %%
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dv = DictVectorizer(sparse=False)
X_val = dv.fit_transform(val_dicts)

# %%
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %%
importances = dt.feature_importances_
feature_names = dv.feature_names_

feature_importance = list(zip(feature_names, importances))
sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

sorted_importance


# %%

features = [x[0] for x in sorted_importance]
importances = [x[1] for x in sorted_importance]

plt.figure(figsize=(6, 4))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### C. Model selection process and parameter tuning

# %% [markdown]
# ### C-1. Decision Tree Model

# %% [markdown]
# #### selected features

# %%
selected_features = [name for name, importance in feature_importance if importance > 0.05]

print(selected_features)

selected_indices = [feature_names.index(name) for name in selected_features]

X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# %% [markdown]
# #### Parameter Tuning

# %%
# max_depth

depths = [1,2,3,4,5,6,10,15,20, None]

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_selected, y_train)
    
    y_pred_val = dt.predict_proba(X_val_selected)[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    
    print(f'Depth {depth}: Validation AUC = {auc_val:.3f}')
    

# %%
# min_samples_leaf
scores = []

for depth in [4,5,6]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train_selected, y_train)
        
        y_pred = dt.predict_proba(X_val_selected)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((depth, s, auc))

# %%
columns = ['max_depth', 'min_sample_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index = 'min_sample_leaf', columns='max_depth', values = 'auc')
df_scores_pivot.round(3)

# %%
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
plt.show()

# %%
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train_selected, y_train)

print(export_text(dt, feature_names=selected_features))

# %% [markdown]
# ### C-2. Random Forest Model

# %%
scores = []

for n in range(10,201,10):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train_selected, y_train)
    
    y_pred = rf.predict_proba(X_val_selected)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    scores.append((n, auc))

# %%
df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores

# %%
plt.plot(df_scores.n_estimators, df_scores.auc)
plt.show()

# %%
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        rf.fit(X_train_selected, y_train)
        
        y_pred = rf.predict_proba(X_val_selected)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((d, n, auc))

# %%
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores

# %%
for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc, label='max_depth=%d' % d)
    
plt.legend()
plt.show()

# %%
max_depth = 15

# %%
scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=s, random_state=42)
        rf.fit(X_train_selected, y_train)
        
        y_pred = rf.predict_proba(X_val_selected)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((s, n, auc))

# %%
columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores

# %%
colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    plt.plot(df_subset.n_estimators, df_subset.auc, color=col, label='min_samples_leaf=%d' % s)
    
plt.legend()
plt.show()

# %%
params = {
    'n_estimators': [100, 110, 120],
    'max_depth': [max_depth],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=params, scoring='roc_auc', cv=3, verbose=2)
grid_search.fit(X_train_selected, y_train)

best_rf = grid_search.best_estimator_


# %%
print("Best rf: ", best_rf)

# %%
rf = RandomForestClassifier(n_estimators=110, max_depth=15, min_samples_leaf=5, random_state=42)
rf.fit(X_train_selected, y_train)

y_pred = rf.predict_proba(X_val_selected)[:, 1]
rf_val_auc = roc_auc_score(y_val, y_pred)

print("RF Val AUC: ", rf_val_auc)

# %% [markdown]
# #### Select the final model

# %%
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train_selected, y_train)

y_pred = dt.predict_proba(X_val_selected)[:, 1]
dt_auc= roc_auc_score(y_val, y_pred)

print("Decision_Tree AUC: ", dt_auc)

# %%
rf = RandomForestClassifier(n_estimators=110, max_depth=15, min_samples_leaf=5, random_state=42)
rf.fit(X_train_selected, y_train)

y_pred = rf.predict_proba(X_val_selected)[:, 1]
rf_val_auc = roc_auc_score(y_val, y_pred)

print("Random_Forest AUC: ", rf_val_auc)


