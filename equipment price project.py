#!/usr/bin/env python
# coding: utf-8

# # Adisa Mumeen Suleiman
# In this unsupervised learning project with scikit learn, I used the RandonForestRegressor estimator to predict the sale price of buldozer,
# 
# And the metric used for scoring evaluation is Root Mean Square Log Error, RMSLE
# 
# ## 1. Problem defition
# 
# ** With what accuracy can i predict the future sale price of an equipment,bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?**
# 
# ## 2. Data
# 
# Downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/c/bluebook-for-bulldozers/data
# 
# There are 3 main datasets:
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# ## 3. Evaluation
# 
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation
# 
# **Note:** The goal for most regression evaluation metrics is to minimize the error. For example, my goal for this project will be to build a machine learning model which minimises RMSLE.
# 
# ## 4. Features
# 
# Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[35]:


# Import training and validation sets
buldozer = pd.read_csv("bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False)
buldozer.head(6)


# In[36]:


buldozer.info(); # information about dataset


# In[37]:


buldozer.describe() # statistical description of data


# In[38]:


buldozer.isna().sum() # sum all NAN entries of each feature/attribute/column


# In[39]:


buldozer.columns # list of all columns/features in the dataset


# In[40]:


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(buldozer["saledate"][:1000], buldozer["SalePrice"][:1000], color="Red"); # graphical illustration of the relationship between the saledate and saleprice column


# In[41]:


buldozer.saledate[:1000] # the date format


# In[42]:


buldozer.saledate.dtype # the date is of type object


# In[43]:


buldozer.SalePrice.plot.hist(figsize = (4,3), color = "Hotpink"); #histogram of the salesprice


# ### Parsing dates
# 
# When we work with time series data, we want to enrich the time & date component as much as possible.
# 
# This is important for modeling, as the datatype must be amenable.
# 
# We can do that by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

# In[44]:


# Reimport data, while the date is parsed along
buldozer = pd.read_csv("bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])
buldozer.head()


# In[46]:


buldozer.saledate.dtype


# In[45]:


buldozer.saledate[:1000]


# In[50]:


fig, ax = plt.subplots(figsize =(8,4))
ax.scatter(buldozer["saledate"][:1000], buldozer["SalePrice"][:1000], color = "Green"); # plot of the parsed date with price


# In[51]:


buldozer.head()


# In[52]:


buldozer.head().T # transpose of sales the dataframe


# In[53]:


buldozer.saledate.head(20) # the format now differs after parsing it


# In[54]:


buldozer.saledate.dtype # but still of same dtype without parsing it


# ### Sort DataFrame by saledate
# 
# When working with time series data, it's a good idea to sort it by date.

# In[55]:


# Sort DataFrame in date order
buldozer.sort_values(by=["saledate"], inplace=True, ascending=True)
buldozer.saledate.head(20)


# In[56]:


# Make a copy of the original DataFrame to perform edits on
buldozer_copy= buldozer.copy()


# ### Add datetime parameters for `saledate` column

# In[62]:


buldozer_copy["saleYear"] = buldozer_copy.saledate.dt.year
buldozer_copy["saleMonth"] = buldozer_copy.saledate.dt.month
buldozer_copy["saleDay"] = buldozer_copy.saledate.dt.day
buldozer_copy["saleDayOfWeek"] = buldozer_copy.saledate.dt.dayofweek
buldozer_copy["saleDayOfYear"] = buldozer_copy.saledate.dt.dayofyear


# In[58]:


buldozer_copy.head().T


# In[61]:


# Now we've enriched our DataFrame with date time features, we can remove 'saledate'
buldozer_copy.drop("saledate", axis=1, inplace=True)
buldozer_copy # saledate column has removed and sale-syear/month/days have been added


# In[63]:


buldozer_copy


# In[64]:


# Check the values of different states on the state columns
buldozer_copy.state.value_counts()


# In[65]:


len(buldozer_copy) # there are 412698 rows on the buldozer copy dataset


# ## 5. Modelling-Driven EDA
# 
# We've done some Exploartory Data Analysis, but let's start to do some model-driven Exploratory Data Analysis.

# In[73]:


# Let's build a machine learning model 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

model = RandomForestRegressor(n_jobs=-1,
                              random_state=42)

model.fit(buldozer_copy.drop("SalePrice", axis=1), buldozer_copy["SalePrice"]) # this will not work because there are missing  and object data in the dataset


# In[72]:


buldozer_copy.info();


# In[74]:


buldozer_copy["UsageBand"].dtype # for example the UsageBand Column contains object datatype


# In[75]:


buldozer_copy.isna().sum() # check the numbers of missing data


# ### Convert string to categories
# 
# One way we can turn all of our data into numbers is by converting them into pandas catgories.
# 
# We can check the different datatypes compatible with pandas here: https://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#data-types-related-functionality

# In[77]:


buldozer_copy.head().T


# In[78]:


pd.api.types.is_string_dtype(buldozer_copy["state"])


# In[79]:


# Find all the columns which contain strings
for label, content in buldozer_copy.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[80]:


# This will turn all of the string value into category values
for label, content in buldozer_copy.items():
    if pd.api.types.is_string_dtype(content):
        buldozer_copy[label] = content.astype("category").cat.as_ordered()


# In[82]:


buldozer_copy.info();


# In[83]:


buldozer_copy.state.cat.categories # the state content have been converted to category type


# In[85]:


buldozer_copy.state.cat.codes # assigned with diffetent category codes


# Thanks to pandas Categories we now have a way to access all of our data in the form of numbers.
# 
# But we still have a bunch of missing data...

# In[86]:


# Check missing data
buldozer_copy.isnull().sum()


# ### Save preprocessed data

# In[87]:


# Export current buldozer dataframe
buldozer_copy.to_csv("bluebook-for-bulldozers/train_copy.csv",
              index=False)


# In[88]:


# Reimport preprocessed data
buldozer_copy = pd.read_csv("bluebook-for-bulldozers/train_copy.csv",
                     low_memory=False)
buldozer_copy.head().T


# In[89]:


buldozer_copy.isna().sum() #dont forget, there are still missing data. so modelling wont work yet


# ## Fill missing values 
# 
# ### Fill numerical missing values first

# In[91]:


for label, content in buldozer_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label) # lets see all the numeric content of our dataset


# In[92]:


buldozer_copy.ModelID # example is the ModelID


# In[95]:


# Check for which numeric columns have null values OUT of all the numeric columns
for label, content in buldozer_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[98]:


# Fill each numeric missing rows with the median of each column
for label, content in buldozer_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary(True or false) column which tells us if the data was missing or not
            buldozer_copy[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            buldozer_copy[label] = content.fillna(content.median())
buldozer_copy


# In[97]:


# Check if there's any null numeric values
for label, content in buldozer_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[99]:


# Check to see how many examples were missing
buldozer_copy.auctioneerID_is_missing.value_counts()


# In[102]:


buldozer_copy.isna().sum() # numeric clumns with missing values has been filled, lets work on non-numeric


# ### Filling and turning categorical variables into numbers

# In[103]:


# Check for columns which aren't numeric
for label, content in buldozer_copy.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[104]:


# Turn categorical variables into numbers and fill missing
for label, content in buldozer_copy.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        buldozer_copy[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        buldozer_copy[label] = pd.Categorical(content).codes+1


# In[105]:


pd.Categorical(buldozer_copy["state"]).codes+1


# In[106]:


buldozer_copy.info()


# In[107]:


buldozer_copy.head().T


# In[108]:


buldozer_copy.isna().sum() # wow!!! amazing! no more missing value in the dataset, all done
# the numeric columns filled with median, while the non-numeric column filled with catgorical codes


# Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# In[110]:


buldozer_copy.head() # lets view it


# In[111]:


len(buldozer_copy)


# In[112]:


get_ipython().run_cell_magic('time', '', '# Instantiate model\nmodel = RandomForestRegressor(n_jobs=-1,\n                              random_state=42)\n\n# Fit the model\nmodel.fit(buldozer_copy.drop("SalePrice", axis=1), buldozer_copy["SalePrice"])\n')


# In[113]:


# Score the model
model.score(buldozer_copy.drop("SalePrice", axis=1), buldozer_copy["SalePrice"])


# **Question:** Why doesn't the above metric hold water? (why isn't the metric reliable
# 
# This does not make the sense we want and not reliable becuase, the dataset has not been splited into train and test data.

# ### Splitting data into train/validation sets

# In[114]:


buldozer_copy.saleYear


# In[115]:


buldozer_copy.saleYear.value_counts()


# In[118]:


# Split data into training and validation
buldozer_copy_validation = buldozer_copy[buldozer_copy.saleYear == 2012]
buldozer_copy_train = buldozer_copy[buldozer_copy.saleYear != 2012]

len(buldozer_copy_validation), len(buldozer_copy_train)


# In[120]:


# Split data into X & y
X_train, y_train = buldozer_copy_train.drop("SalePrice", axis=1), buldozer_copy_train.SalePrice
X_valid, y_valid = buldozer_copy_validation.drop("SalePrice", axis=1), buldozer_copy_validation.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[97]:


y_train


# ### Building an evaluation function

# In[121]:


# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores


# ## Testing our model on a subset (to tune the hyperparameters)

# In[123]:


# # This takes far too long... for experimenting

get_ipython().run_line_magic('time', '')
model = RandomForestRegressor(n_jobs=-1, 
                              random_state=42)

model.fit(X_train, y_train)


# In[101]:


len(X_train)


# In[108]:


# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)


# In[124]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time\nmodel.fit(X_train, y_train)\n')


# In[125]:


show_scores(model)


# ### Hyerparameter tuning with RandomizedSearchCV

# In[126]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n# Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators": np.arange(10, 100, 10),\n           "max_depth": [None, 3, 5, 10],\n           "min_samples_split": np.arange(2, 20, 2),\n           "min_samples_leaf": np.arange(1, 20, 2),\n           "max_features": [0.5, 1, "sqrt", "auto"],\n           "max_samples": [10000]}\n\n# Instantiate RandomizedSearchCV model\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                    random_state=42),\n                              param_distributions=rf_grid,\n                              n_iter=2,\n                              cv=5,\n                              verbose=True)\n\n# Fit the RandomizedSearchCV model\nrs_model.fit(X_train, y_train)\n')


# In[127]:


# Find the best model hyperparameters
rs_model.best_params_


# In[128]:


# Evaluate the RandomizedSearch model
show_scores(rs_model)


# ### Train a model with the best hyperparamters
# 
# **Note:** These were found after 100 iterations of `RandomizedSearchCV`.

# In[129]:


get_ipython().run_cell_magic('time', '', '\n# Most ideal hyperparamters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs=-1,\n                                    max_samples=None,\n                                    random_state=42) # random state so our results are reproducible\n\n# Fit the ideal model\nideal_model.fit(X_train, y_train)\n')


# In[130]:


# Scores for ideal_model (trained on all the data)
show_scores(ideal_model)


# In[131]:


# Scores on rs_model (only trained on ~10,000 examples)
show_scores(rs_model)


# ## Make predictions on test data

# In[ ]:


# Import the test data
buldozer_test = pd.read_csv("bluebook-for-bulldozers/Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])

buldozer_test.head()


# In[ ]:


# Make predictions on the test dataset
test_preds = ideal_model.predict(buldozer_test)


# ### Preprocessing the data (getting the test dataset in the same format as our training dataset)

# In[141]:


def preprocess_data(buldozer):
    """
    Performs transformations on df and returns transformed df.
    """
    buldozer["saleYear"] = buldozer.saledate.dt.year
    buldozer["saleMonth"] = buldozer.saledate.dt.month
    buldozer["saleDay"] = buldozer.saledate.dt.day
    buldozer["saleDayOfWeek"] = buldozer.saledate.dt.dayofweek
    buldozer["saleDayOfYear"] = buldozer.saledate.dt.dayofyear
    
    buldozer.drop("saledate", axis=1, inplace=True)
    
    # Fill the numeric rows with median
    for label, content in buldozer.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                buldozer[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                buldozer[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            buldozer[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            buldozer[label] = pd.Categorical(content).codes+1
    
    return buldozer


# In[133]:


# Process the test data 
buldozer_test = preprocess_data(buldozer_test)
buldozer_test.head()


# In[ ]:


# Make predictions on updated test data
test_preds = ideal_model.predict(buldozer_test)


# In[ ]:


X_train.head()


# In[ ]:


# We can find how the columns differ using sets
set(X_train.columns) - set(buldozer_test.columns)


# In[ ]:


# Manually adjust df_test to have auctioneerID_is_missing column
buldozer_test["auctioneerID_is_missing"] = False
buldozer_test.head()


# Finally now our test dataframe has the same features as our training dataframe, we can make predictions!

# In[ ]:


# Make predictions on the test data
test_preds = ideal_model.predict(buldozer_test)


# In[152]:


test_preds


# We've made some predictions but they're not in the same format Kaggle is asking for: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

# In[153]:


# Format predictions into the same format Kaggle is after
buldozer_preds = pd.DataFrame()
buldozer_preds["SalesID"] = buldozer_test["SalesID"]
buldozer["SalesPrice"] = test_preds
buldozer_preds


# In[ ]:


# Export prediction data
buldozer_preds.to_csv("bluebook-for-bulldozers/test_predictions.csv", index=False)


# ### Feature Importance
# 
# Feature importance seeks to figure out which different attributes of the data were most importance when it comes to predicting the **target variable** (SalePrice).

# In[ ]:


# Find feature importance of our best model
ideal_model.feature_importances_


# In[138]:


# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    buldozer = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots(figsize= (6,4))
    ax.barh(buldozer["features"][:n], buldozer["feature_importances"][:20], color="Green")
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[139]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[137]:


buldozer["Enclosure"].value_counts()

