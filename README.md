# 🏠 Real Estate Price Prediction – Bargain Project (2024)

This project aims to predict whether a real estate property in France was sold for more than €350,000 using property characteristics, geographic information, and socio-economic municipal data sourced from Open Data.

## 📌 Objective

Build a classification model to determine whether a property surpasses the €350,000 sale price threshold. The project emphasizes feature enrichment using public datasets and advanced preprocessing techniques to boost model performance.

## 📊 Datasets

- **DVF (Demandes de Valeurs Foncières)** — real estate transaction data (2021–2022)
- **INSEE** — income and demographic data per municipality
- **IGN GeoJSON** — municipal and departmental geometries
- **Densité INSEE** — rural/urban classification
- **Custom dataset** — includes geolocated real estate properties and target variable (`sup350k`)

All data is merged using geographic coordinates and INSEE municipal codes.

## ⚙️ Key Features

- **Geospatial data integration**: Assigns each property to a municipality using spatial joins and nearest-neighbor correction for coastal/missing properties.
- **Municipality-level feature engineering**: Income median, population density, price per square meter evolution, etc.
- **Missing data imputation**: KNN-based imputation for numerical features.
- **Custom ratios**: Surface per room, surface per bathroom, household ratios, etc.
- **Modeling**: Random Forest, CatBoost, XGBoost, and Stacking Classifiers.

## 🧠 Models & Results

| Model              | ROC-AUC Score |
|-------------------|---------------|
| CatBoost           | ~0.96         |
| Random Forest      | ~0.975        |
| Stacking Ensemble  | 0.976         |
| XGBoost            | **0.98**      |


## 📈 Feature Importance (Top 10)

- Property size compared to the average size in the area **(added)** 
- Average price per square meter in the municipality **(added)**  
- Average real estate transaction price in the municipality **(added)**   
- Total surface area of the property  
- Energy performance score of the property  
- Average surface per bathroom (computed)   
- Number of bedrooms  
- Median income of households in the municipality **(added)**   
- Total number of rooms in the property  
- Proportion of apartments in the municipality **(added)**   

These top features highlight the importance of local real estate trends, socio-economic context, and intrinsic property characteristics when predicting whether a property exceeds €350,000 in value.

## 🗺️ Visualizations

- Choropleth map of property prices per department
- Heatmap of high-value properties
- Filterable map (Folium) of listings by property type
- Correlation matrix and feature importance bar plots

## 🧰 Tech Stack

- **Python** (pandas, geopandas, scikit-learn, xgboost, catboost, seaborn, matplotlib, folium)
- **Jupyter Notebooks** for exploration and modeling
- **Open Data sources** from INSEE, Etalab, IGN, etc.

## 🧪 Notebooks

- `communes.ipynb`: Data collection and feature engineering using municipal-level data
- `main.ipynb`: Model training, evaluation, and visualization

