
# 🏠 House Price Prediction Project 📊✨

## 🔍 Project Overview

This project focuses on building an interpretable house price prediction model using Python 🐍. It combines Exploratory Data Analysis (EDA), feature engineering, and regression modeling to understand what truly drives house prices — not just prediction accuracy.

📌 The emphasis is on business understanding + model interpretability.

## 🎯 Project Objective

The key objectives of this project are:

💰 Predict house prices using historical housing data

📈 Identify important features influencing house prices

🧠 Apply clean, explainable machine learning techniques

🏘️ Provide insights useful for buyers, sellers, and analysts

## 📊 Dataset Description

The dataset (`HousePrediction.xlsx`) contains:

🏡 Property features: LotArea, TotalBsmtSF, OverallCond

🕒 Time-based features: YearBuilt, YearRemodAdd

🏷️ Categorical features: MSZoning, BldgType, Exterior1st

🎯 Target variable: SalePrice

## 🔎 Exploratory Data Analysis (EDA)

During EDA, the following analyses were performed:

🔍 Checked data types and missing values

📉 Visualized distributions of continuous variables

📊 Used scatter plots and regression plots to study relationships

🧩 Analyzed categorical variables using average SalePrice by category

⚠️ Correctly treated numeric-looking categorical features (e.g., MSSubClass)

📌 Charts shared in the project visually support these findings.

# 📸 Exploratory Data Analysis (EDA) Screenshots

Below are some key visualizations generated during EDA to support findings and interpretations:

### YearBuilt vs SalePrice
![YearBuilt vs SalePrice](images/yearbuilt_vs_saleprice.png)

### YearRemodAdd vs SalePrice
![YearRemodAdd vs SalePrice](images/yearremodadd_vs_saleprice.png)

### LotArea vs SalePrice
![LotArea vs SalePrice](images/lotarea_vs_saleprice.png
)

### TotalBsmtSF vs SalePrice
![TotalBsmtSF vs SalePrice](images/totalbsmtsf_vs_saleprice.png)

### Distribution of Log_LotArea
![Distribution of Log_LotArea](images/log_lotarea_distribution.png
)

### Distribution of Log_TotalBsmtSF
![Distribution of Log_TotalBsmtSF](images/log_totalbsmtsf_distribution.png
)

### Distribution of Log_SalePrice
![Distribution of Log_SalePrice](images/log_saleprice_distribution.png
)

### SalePrice by MSZoning
![SalePrice by MSZoning](images/mszoning_vs_saleprice.png)

### SalePrice by LotConfig
![SalePrice by LotConfig](images/lotconfig_vs_saleprice.png)

### SalePrice by BldgType
![SalePrice by BldgType](images/bldgtype_vs_saleprice.png
)

### SalePrice by Exterior1st
![SalePrice by Exterior1st](images/exterior1st_vs_saleprice.png
)

### SalePrice by MSSubClass
![SalePrice by MSSubClass](images/mssubclass_vs_saleprice.png)

## 🛠️ Feature Engineering

To improve model performance and interpretability:

🔄 Applied log transformation to reduce skewness (LotArea, TotalBsmtSF, SalePrice)

⏳ Created HouseAge and RenovationAge features

🆕 Built a RecentlyRenovated indicator

🔗 Removed redundant features to handle multicollinearity

🧩 Applied one-hot encoding for categorical variables

## 🤖 Modeling Approach

📐 Trained a Linear Regression model on log-transformed SalePrice

📏 Scaled numerical features using StandardScaler

📊 Evaluated model using RMSE and R² score

🔁 Compared results with Ridge Regression for validation

✅ Selected Linear Regression as the final model due to:
	- Better interpretability
	- Stable coefficients
	- Strong explanatory power

## 📈 Key Insights from Charts & Model

✨ Key takeaways from visualizations and coefficients:

⬇️ HouseAge has a strong negative impact on price (depreciation effect)

⬆️ LotArea & TotalBsmtSF significantly increase house prices

🏘️ Zoning and building type create clear price premiums/discounts

🔧 Renovation impact fades as renovation becomes older

📌 These insights are clearly supported by your charts.

## ✅ Conclusion

This project demonstrates how simple models + strong feature engineering can deliver:

- Clear business insights
- Explainable results
- Reliable predictions

🎯 The final Linear Regression model is ideal for analytical and decision-making use cases.

## 🧰 Tools & Technologies

🐍 Python
📦 Pandas, NumPy
📊 Matplotlib, Seaborn
🤖 Scikit-learn
📓 Jupyter Notebook


## Project Structure

```
House Price Prediction.ipynb   # Jupyter notebook for interactive analysis and modeling
House Price Prediction.py      # Main Python script for house price prediction
main.py                       # (Optional) Entry point for running the project
pyproject.toml                # Python project configuration (dependencies, build system)
README.md                     # Project documentation
requirements.txt              # List of required Python packages
```

## Getting Started
1. Install dependencies:
	```
	pip install -r requirements.txt
	```
2. Run the main script:
	```
	python "House Price Prediction.py"
	```
3. For interactive analysis, open `House Price Prediction.ipynb` in Jupyter.

## File Descriptions
- **House Price Prediction.py**: Main script for data loading, preprocessing, model training, and prediction.
- **House Price Prediction.ipynb**: Notebook for exploratory data analysis, visualization, and model experimentation.
- **main.py**: (Optional) Can be used as an entry point for the project.
- **requirements.txt**: Lists required Python packages (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn).
- **pyproject.toml**: Project configuration and metadata.
- **README.md**: Project documentation and usage instructions.

## Conventions
- Follow PEP8 for code style.
- Use modular functions and clear docstrings.
- Place data files in a `data/` directory if needed.
- Save trained models in a `models/` directory if used.

## License
Specify your license here if applicable.
