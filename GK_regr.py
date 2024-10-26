import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


'''Setup variables'''
# Specify the file path
file_path = 'GK_data.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Filter the DataFrame for missing values
cleaned_df = df.dropna()

# Select the second column as Y values
Y = cleaned_df.iloc[:, 1]

# Select all columns from the third onwards as X values
X = cleaned_df.iloc[:, 2:]

'''Sequential Feature Selector'''
# Initialize the model (e.g., Linear Regression)
model = LinearRegression()

# Initialize Sequential Feature Selector (SFS)
sfs = SFS(model, 
        n_features_to_select='auto',  # Choose the number of features you want to select
        direction='forward')     # Can be 'forward' or 'backward'

# Fit SFS to the training data
sfs.fit(X, Y)

# Get the selected features
selected_features = X.columns[sfs.get_support()]

# Prepare the data with selected features
X_selected = cleaned_df[selected_features]

'''Correlation Matrix'''
# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(X_selected.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Removing iterably each variable with the highest VIF until all are <10
X_correlation_checked = X_selected.drop(columns=[
    'SoTA','PSxG','Saves','PSxG/SoT','Launch%','Thr','AvgDist','L']
    )
print(X_correlation_checked)

plt.figure(figsize=(16, 12))
sns.heatmap(X_correlation_checked.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

'''VIF'''
# VIF dataframe
vif_data = pd.DataFrame()
vif_data['Feature'] = X_correlation_checked.columns
print(vif_data.head())

# Calculate VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X_correlation_checked.values, i) 
                   for i in range(len(X_correlation_checked.columns))]
print(vif_data)

'''Model Summary'''
# Add constant for the intercept
X_correlation_checked = sm.add_constant(X_correlation_checked)
print('constant', X_correlation_checked)
model = sm.OLS(Y, X_correlation_checked).fit()
print(model.summary())

'''Train-Test split'''
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_correlation_checked, Y, test_size=1/10)

train_test_model = LinearRegression()
train_test_model.fit(X_train, Y_train)
result = train_test_model.score(X_test, Y_test)
print("R2:", result)

'''Cross-Validation'''
# Perform cross-validation
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(train_test_model, X_selected, Y, cv=kfold)
print("R2:", results.mean(), 'std:', results.std(), 'R2s:', results)

