# House Price Prediction Script
"""
This script performs house price prediction using data analysis, feature engineering, and linear/ridge regression modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Set a modern, attractive style for all charts
sb.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 4),
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

#Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Data Overview Function
def data_overview(df):
    print('Shape:', df.shape)
    print('Columns:', df.columns)
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.tail())
    print('Nulls in SalePrice:', df['SalePrice'].isnull().sum())

# Load Dataset
df = pd.read_excel('HousePrediction.xlsx')
# Drop rows with null SalePrice immediately
df = df.dropna(subset=['SalePrice'])
data_overview(df)
data_overview(df)



# Feature Counts
num_count = df.select_dtypes(include=['number']).shape[1]
cat_count = df.select_dtypes(include=['object']).shape[1]
print(f'Numerical_features_Count: {num_count}')
print(f'Categorical_feature_Count: {cat_count}')

# Feature Lists
year_features = ['YearBuilt', 'YearRemodAdd']
Continuous_Features = ['LotArea', 'BsmtFinSF2', 'TotalBsmtSF']
Cat_features = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st', 'MSSubClass']

# Categorical Feature Analysis
for col in Cat_features:
    print(f"\n{col}")
    print(df.groupby(col)['SalePrice'].mean().sort_values(ascending=False))

# Boxplots for categorical features with vibrant color palette
for col in Cat_features:
    plt.figure(figsize=(8,4))
    sb.boxplot(x=col, y='SalePrice', data=df, palette="Set3")
    plt.xticks(rotation=45)
    plt.title(f'SalePrice by {col}', fontsize=15, fontweight='bold')
    plt.xlabel(col, fontsize=13)
    plt.ylabel('SalePrice', fontsize=13)
    plt.tight_layout()
    plt.show()

# Correlation checks
def correlation_analysis(df):
    for col in year_features:
        corr = df[col].corr(df['SalePrice'])
        print(f'Correlation between {col} and SalePrice: {corr:.4f}')
    print('OverallCond vs SalePrice (spearman):', df['OverallCond'].corr(df['SalePrice'], method='spearman'))
    for col in Continuous_Features:
        corr = df[col].corr(df['SalePrice'])
        print(f'Correlation between {col} and SalePrice: {corr:.4f}')

correlation_analysis(df)

# Histograms for continuous features with attractive colors
# 'husl' palette gives a rainbow effect; you can use 'coolwarm', 'viridis', or any matplotlib colormap
for i, col in enumerate(Continuous_Features):
    plt.figure(figsize=(8,4))
    sb.histplot(df[col], bins=38, kde=True, color=sb.color_palette("husl", len(Continuous_Features))[i])
    plt.title(f'Distribution of {col}', fontsize=15, fontweight='bold')
    plt.xlabel(f'{col} (sq ft)', fontsize=13)
    plt.ylabel('Number of Houses', fontsize=13)
    plt.tight_layout()
    plt.show()

# SalePrice distribution with a vibrant color
plt.figure(figsize=(8,4))
sb.histplot(df['SalePrice'], bins=38, kde=True, color=sb.color_palette("husl", 8)[2])
plt.title('Distribution of SalePrice', fontsize=15, fontweight='bold')
plt.xlabel('SalePrice', fontsize=13)
plt.ylabel('Number of Houses', fontsize=13)
plt.tight_layout()
plt.show()


# Drop feature with high missing/skew
df = df.drop('BsmtFinSF2', axis=1)

# Feature Engineering
# Log transform
for col in ['LotArea', 'TotalBsmtSF', 'SalePrice']:
    df[f'Log_{col}'] = np.log1p(df[col])

# HasBsmt feature
df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)

# House/Renovation Age
df['YrSold'] = df['YearBuilt'].max()
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RenovationAge'] = df['YrSold'] - df['YearRemodAdd']
df['RecentlyRenovated'] = (df['RenovationAge'] <= 10).astype(int)

# Drop year features to avoid multicollinearity
df.drop(['YearBuilt', 'YearRemodAdd', 'YrSold'], axis=1, inplace=True)

# One-hot encode categorical features
df['MSSubClass'] = df['MSSubClass'].astype(str)
cat_cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st', 'MSSubClass']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# Drop unused features
df_encoded = df_encoded.drop(['Id', 'LotArea', 'HasBsmt', 'TotalBsmtSF'], axis=1)

# Model Building
X = df_encoded.drop(['SalePrice', 'Log_SalePrice'], axis=1)
y = df_encoded['Log_SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)
Y_pred_log = lr_model.predict(X_test_scaled)
Y_pred_Price = np.expm1(Y_pred_log)
Y_test_Price = np.expm1(Y_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_log))
r2 = r2_score(Y_test, Y_pred_log)
print('Linear Regression RMSE (log):', rmse)
print('Linear Regression R2:', r2)

# Coefficient interpretation
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_}).sort_values(by='Coefficient', ascending=False)
print('Top 10 Coefficients:')
print(coef_df.head(10))
print('Bottom 10 Coefficients:')
print(coef_df.tail(10))

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, Y_train)
y_ridge_pred = ridge_model.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(Y_test, y_ridge_pred))
r2_score_ridge = r2_score(Y_test, y_ridge_pred)
print('Ridge Regression RMSE (log):', rmse_ridge)
print('Ridge Regression R2:', r2_score_ridge)

# Model Comparison
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression'],
    'RMSE (log)': [rmse, rmse_ridge],
    'R2 Score': [r2, r2_score_ridge]
})
print(comparison_df)

