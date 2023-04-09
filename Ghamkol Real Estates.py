#!/usr/bin/env python
# coding: utf-8

# ## Ghamkol Real Estate Price Predictor

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing['ZN'].value_counts()


# In[5]:


# # for plotting histogram
# housing.hist(bins = 50, figsize = (20,15))


# ## Train-test splitting

# In[6]:


# # For learning purpose
# import numpy as np

# def split_train_test(data, test_ratio):
#     np.random.seed(42) # used so testing and training values dont overlap
#     shuffled = np. random.permutation(len(data))
#     # print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[7]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[8]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[9]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[10]:


# There is a possibility that CHAS can be one and that the training 
# data set cant catch that, thus we use stratified learning
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[11]:


strat_test_set['CHAS'].value_counts()


# In[12]:


strat_train_set['CHAS'].value_counts()


# In[13]:


housing = strat_train_set.copy()


# ## Looking for correlations

# In[14]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[15]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[16]:


housing.plot(kind = "scatter", x = "RM", y = "MEDV", alpha = 0.8)


# ## Trying out attribute combinations

# In[17]:


# used to make the data better, as understanding the data is very important
# TAXRM, used to collect tax/RM

housing['TAXRM'] = housing["TAX"]/housing["RM"]


# In[18]:


housing.head()


# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


housing.plot(kind = "scatter", x = "TAXRM", y = "MEDV", alpha = 0.8)


# In[21]:


# segregating features and labels
housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing attributes

# In[22]:


# to take care of missing attributes, I have 3 options:
#     1. To get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value (0, mean or median)


# In[23]:


# This is for missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[24]:


X = imputer.transform(housing) # fills any missing data


# ## Scikit-learn Design

# Primarily, 3 types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. Imputer. It has a fit method and a transform method. fit method: Fits the dataset abd calculates internal parameters.
# 
# 2. Transformers - Transform method takes input and returns output based on the learnings from fit(). It also has a convienience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predic() are two common functions. It also gives score function which will evaluate the predictions.

# ## Feature scaling

# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
# (value - min)/(max - min)
# sklearn provides a class called MinMaxScaler for this
#     
# 2. Standardization
# (value - mean)/sd 
# sd: standard deviation
# variance becomes 1
# sklearn provides a class called Standard Scaler for this
#     

# ## Making the pipeline : Pipeline ka matlab automate karna, series of steps karna

# In[25]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('std_scaler', StandardScaler()),
    # we can add as many as we like
])


# In[26]:


housing_tr = my_pipeline.fit_transform(housing)


# In[27]:


housing_tr.shape


# ## Selecting a desired model for Ghamkol Real Estates

# In[28]:


# testing different models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_tr, housing_labels)


# In[29]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[30]:


# checking our predictions
prepared_data = my_pipeline. transform(some_data)


# In[31]:


model.predict(prepared_data)


# In[32]:


list(some_labels)


# ## Evaluating the model

# In[33]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[34]:


rmse 
# this is 4.829 for Linear Regression, too much so I will try another model
# its 0 for DTR, so overfitting occurs


# ## Using better evaluation technique - Cross validation

# In[35]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


# In[36]:


rmse_scores


# In[37]:


# decision tree regressor works better so thats the one i chose
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[38]:


print_scores(rmse_scores)


#  ## Saving the model

# In[39]:


from joblib import dump, load
dump(model, 'Ghamkol.joblib')


# ## Testing the model on Test data

# In[42]:


X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_predictions, list(Y_test))
# In a very small dataset, this works very well.


# In[41]:


final_rmse
# 2.93, model is working well


# I did not fine tune models, hoping to do that in the future. No one can guarantee which model will work better so you have to try all the models. 

# ## Using the model

# In[45]:


from joblib import dump, load
import numpy as np
model = load('Ghamkol.joblib')
features = np.array([[-0.44352175,  3.12628155, -1.35893781, -0.27288841, -1.0542567 ,
        0.49865392, -1.3938808 ,  2.19312325, -0.65766683, -0.78557904,
       -0.69277865,  0.39131918, -0.94116739]])
model.predict(features) 
# this is the predicted price
