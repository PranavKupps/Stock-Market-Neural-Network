# Import packages
# You need to install the package scikit-learn in order for lines referring to sklearn to run
# You need to install the package python-docx in order for lines referring to docx to run
import pandas as pd
import statsmodels.api as sm
import numpy as np
import datetime
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestRegressor

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Import and view data
returns01 = pd.read_csv('Final data 20250312_2300.csv')
returns01['month'] = pd.to_datetime(returns01['month'], format='%d-%b-%Y')
returns01 = returns01[returns01['g07epadj'].notna()]

datecut1 = datetime.datetime(2016, 1, 31)
datecut2 = datetime.datetime(2023, 1, 31)

returns01_train   = returns01[ returns01['month'] <= datecut1]
returns01_valid   = returns01[(returns01['month'] >  datecut1) & (returns01['month'] < datecut2)]
returns01_all     = returns01[ returns01['month'] <  datecut2]
returns01_predict = returns01[ returns01['month'] == datecut2]
print(returns01_train.shape)
print(returns01_valid.shape)
print(returns01_all.shape)
print(returns01_predict.shape)

depvar = ['g07epadj']
indvar = [                              'lag1mcreal',
                'finlag1bmmiss',        'finlag1bm',
                                        'fing01dyadj',
                'fing02esgmiss',        'fing02esg',
                'fing03nibadjmiss',     'fing03nibadj',
                'fing04fcfyadjmiss',    'fing04fcfyadj',
                'fing05rdsadjmiss',     'fing05rdsadj',
                'fing08sadadjmiss',     'fing08sadadj',
                'fing09shoadjmiss',     'fing09shoadj',
                'fing10shiadjmiss',     'fing10shiadj',
                'fing11ret5adjmiss',    'fing11ret5adj',
                'fing12empadjmiss',     'fing12empadj',
                'fing13sueadjmiss',     'fing13sueadj',
                'fing14erevadjmiss',    'fing14erevadj']

y_train = returns01_train[depvar]
x_train = returns01_train[indvar]
y_valid = returns01_valid[depvar]
x_valid = returns01_valid[indvar]
y_all = returns01_all[depvar]
x_all = returns01_all[indvar]
y_predict = returns01_predict[depvar]
x_predict = returns01_predict[indvar]

print(x_train.shape)
print(y_train.shape)
print(x_train.head(10))
print(y_train.head(10))

x_train_array   = x_train.values
y_train_array   = y_train.values
y_train_array   = y_train_array.ravel()
x_valid_array   = x_valid.values
y_valid_array   = y_valid.values
y_valid_array   = y_valid_array.ravel()
x_all_array     = x_all.values
y_all_array     = y_all.values
y_all_array     = y_all_array.ravel()
x_predict_array = x_predict.values
y_predict_array = y_predict.values
y_predict_array = y_predict_array.ravel()

# First, consider training and validation sets and determine hyperparameters

# Specify hyperparameters
rf = RandomForestRegressor(max_depth=3, min_weight_fraction_leaf=0.08, n_estimators=100, random_state=11610,
                           max_features=7, bootstrap=True, max_samples=0.50)

# Run the model on training data and see how well it performs in validation
rfmodel_train = rf.fit(x_train_array, y_train_array)

rfresult_train = rf.score(x_train_array, y_train_array)
rfscore_train = pd.DataFrame([rfresult_train])
rfimportances_train = rf.feature_importances_
rfsorted_index_train = np.argsort(rfimportances_train)[::-1]
rflabels_train = np.array(indvar)[rfsorted_index_train]
rfimportance_train = pd.DataFrame(list(zip(rflabels_train, rfimportances_train[rfsorted_index_train])), columns=['indvar', 'importances'])

rfpredictions_train = rf.predict(x_train_array)
rfpredictions_valid = rf.predict(x_valid_array)

ssr_train = np.sum((y_train_array - rfpredictions_train)**2)
sst_train = np.sum((y_train_array - np.mean(y_train_array))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train_array - rfpredictions_train)**2))

ssr_valid = np.sum((y_valid_array - rfpredictions_valid)**2)
sst_valid = np.sum((y_valid_array - np.mean(y_valid_array))**2)
rsq_valid = 1 - (ssr_valid/sst_valid)
rmse_valid = np.sqrt(np.mean((y_valid_array - rfpredictions_valid)**2))

print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')
print(f'R-squared in training sample from rsquared function: {rfresult_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')

print(f'Sum of squared difference between y values and predicted y values in validation sample (SSR): {ssr_valid:.5f}')
print(f'Sum of squared difference between y values and average y values in validation sample (SST): {sst_valid:.5f}')
print(f'Square root of the mean squared error in validation sample: {rmse_valid:.5f}')
print(f'R-squared in validation sample = 1 - SSR/SST: {rsq_valid:.5f}')

print(rfimportance_train)

# Second, use all the data excluding January 2023 to form predictions for January 2023
# This section also includes code to generate importances and scores for each tree
# Prepare an empty dataframes to store feature importances
importancesx_all = pd.DataFrame()
# Prepare an empty list of scores
scores_list_all = []

# Specify hyperparameters
rf2 = RandomForestRegressor(max_depth=3, min_weight_fraction_leaf=0.08, n_estimators=100, random_state=11610,
                           max_features=7, bootstrap=True, max_samples=0.50)

# Run the model on training data and see how well it performs in validation
rfmodel_valid = rf2.fit(x_all_array, y_all_array)

rfresult_all = rf2.score(x_all_array, y_all_array)
rfscore_all = pd.DataFrame([rfresult_all])
rfimportances_all = rf2.feature_importances_
rfsorted_index_all = np.argsort(rfimportances_all)[::-1]
rflabels_all = np.array(indvar)[rfsorted_index_all]
rfimportance_all = pd.DataFrame(list(zip(rflabels_all, rfimportances_all[rfsorted_index_all])), columns=['indvar', 'importances'])

rfpredictions_all = rf2.predict(x_all_array)
rfpredictions_predict = rf2.predict(x_predict_array)

ssr_all = np.sum((y_all_array - rfpredictions_all)**2)
sst_all = np.sum((y_all_array - np.mean(y_all_array))**2)
rsq_all = 1 - (ssr_all/sst_all)
rmse_all = np.sqrt(np.mean((y_all_array - rfpredictions_all**2)))

print(f'Sum of squared difference between y values and predicted y values in the full sample excl Jan23 (SSR): {ssr_all:.5f}')
print(f'Sum of squared difference between y values and average y values in the full sample excl Jan23 (SST): {sst_all:.5f}')
print(f'Square root of the mean squared error in the full sample excl Jan23: {rmse_all:.5f}')
print(f'R-squared in the full sample excl Jan23 from rsquared function: {rfresult_all:.5f}')
print(f'R-squared in the full sample excl Jan23 = 1 - SSR/SST: {rsq_all:.5f}')
print(rfimportance_all)

# This block generates importances and scores for each decision tree
for i, tree in enumerate(rf2.estimators_):
    scorex_all = tree.score(x_all_array, y_all_array)
    scores_list_all.append({'Tree': i+1, 'Score': scorex_all})

    temp_df = pd.DataFrame({
        'Feature': x_all.columns,
        'Importance': tree.feature_importances_,
        'Tree': i+1})
    importancesx_all = pd.concat([importancesx_all, temp_df], ignore_index=True)

scores_all = pd.DataFrame(scores_list_all)
scores_all.reset_index(drop=True, inplace=True)
importancesx_all.reset_index(drop=True, inplace=True)
print('Scores by decision tree:',scores_all)

# This block of code combines predicted E/P ratios with the data that went into the model

dfx_all = x_all.rename_axis('oldindex').reset_index()
dfy_all = pd.DataFrame(y_all).rename_axis('oldindex').reset_index()
dfpredictions_all = pd.DataFrame(rfpredictions_all, columns=['pred_ep'])
agg1_all = pd.merge(dfx_all, dfy_all)
print(agg1_all.shape)
agg2_all = agg1_all.merge(dfpredictions_all, how='inner', left_index=True, right_index=True)
print(agg2_all.shape)
selected_columns_all = agg2_all[['oldindex', 'pred_ep']]
returns02_all= returns01_all.merge(selected_columns_all,
    left_index=True, right_on='oldindex')
print(returns02_all.head(10))
print(returns02_all.shape)

dfx_predict = x_predict.rename_axis('oldindex').reset_index()
dfy_predict = pd.DataFrame(y_predict).rename_axis('oldindex').reset_index()
dfpredictions_predict = pd.DataFrame(rfpredictions_predict, columns=['pred_ep'])
agg1_predict = pd.merge(dfx_predict, dfy_predict)
print(agg1_predict.shape)
agg2_predict = agg1_predict.merge(dfpredictions_predict, how='inner', left_index=True, right_index=True)
print(agg2_predict.shape)
selected_columns_predict = agg2_predict[['oldindex', 'pred_ep']]
returns02_predict= returns01_predict.merge(selected_columns_predict, how='inner',
    left_index=True, right_on='oldindex')
print(returns02_predict.head(10))
print(returns02_predict.shape)

# Split the predicted returns from the full sample excluding Jan23 into quintiles and show descriptive statistics by quintile
agg2_all['Quintile'] = pd.qcut(agg2_all['pred_ep'], q=5, labels=[1, 2, 3, 4, 5])

# Group by 'Quintile' and compute summary statistics for 'Variable1' and 'Variable2'
summary_stats = agg2_all.groupby('Quintile',observed=True).agg({
    'g07epadj'       : ['mean', 'median'],
    'pred_ep'        : ['mean', 'median'],
    'lag1mcreal'     : ['mean', 'median'],
    'finlag1bm'      : ['mean', 'median'],
    'fing01dyadj'    : ['mean', 'median'],
    'fing02esg'      : ['mean', 'median'],
    'fing03nibadj'   : ['mean', 'median'],
    'fing04fcfyadj'  : ['mean', 'median'],
    'fing05rdsadj'   : ['mean', 'median'],
    'fing08sadadj'   : ['mean', 'median'],
    'fing09shoadj'   : ['mean', 'median'],
    'fing10shiadj'   : ['mean', 'median'],
    'fing11ret5adj'  : ['mean', 'median'],
    'fing12empadj'   : ['mean', 'median'],
    'fing13sueadj'   : ['mean', 'median'],
    'fing14erevadj'  : ['mean', 'median']
})

print(summary_stats)

# Export summary statistics to Excel
dfrsq = pd.DataFrame({'rsq_train': [rsq_train], 'rsq_valid': [rsq_valid], 'rsq_all': [rsq_all] })
with pd.ExcelWriter('Earnings to price output 20250407_2026.xlsx') as writer:
    dfrsq.to_excel(writer, sheet_name='dfrsq')
    rfimportance_all.to_excel(writer, sheet_name='rfimportance_all')
    scores_all.to_excel(writer, sheet_name='scores_all')
    importancesx_all.to_excel(writer, sheet_name='importancesx_all')
    summary_stats.to_excel(writer, sheet_name='summary_stats')
    returns02_predict.to_excel(writer, sheet_name='returns03_predict')
