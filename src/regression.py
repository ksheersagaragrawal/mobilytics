#This file has the code to implement random forests and xgboost for regression to understand feature importance for target columns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap

from data_preprocessing import read_dataset, RANDOM_STATE, PROJECT_ROOT
from visualization import create_all_regression_visualizations

def train_random_forest(df):
    """
    Train Random Forest model for battery drain prediction.
    """

    # Taking battery drain as a target to understand how battery drains based on numeric and categorical features (one hot encoded)
    target_col = "Battery Drain (mAh/day)"

    numeric_features = ["App Usage Time (min/day)", "Screen On Time (hours/day)", "Data Usage (MB/day)", "Number of Apps Installed", "Age"]

    categorical_features = ["Device Model", "Operating System", "Gender"]

    # Creating input and output (battery drain is what we need to predict, hence it is output)
    X = df[numeric_features + categorical_features]. copy()
    y = df[target_col].copy()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model. fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculating metrics for regression (MAE and MSE are common metrics used) to see how the random forest has worked. 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)

    print(f"MAE : {mae:.2f} mAh/day")
    print(f"RMSE: {rmse:.2f} mAh/day")

    # We defined the model as a pipeline hence taking the random forest stage and the preprocess phase from the pipeline
    rf_model = model.named_steps["rf"]
    pre = model.named_steps["preprocess"]

    # Categorical feature names (after one-hot).  Gives all the unique categories. 
    one_hot_encoder = pre.named_transformers_["cat"]
    one_hot_encoder = one_hot_encoder.get_feature_names_out(categorical_features)

    feature_names = np.concatenate([numeric_features, one_hot_encoder])

    # Getting the importance of each feature as compared to original output column (battery drain)
    importance_of_each_feature = rf_model.feature_importances_

    # Converting the features to series so that we can plot it easily. 
    feat_imp = pd.Series(importance_of_each_feature * 100, index=feature_names). sort_values(ascending=False)

    # Showing the same without a plot and as a table
    print(feat_imp.head(15))

    # Taking a small sample of data (350 features) randomly to get the global importance summary (Shap is slow and few samples should be enough.  Took almost 50 percent)
    X_sample = X_train.sample(350, random_state=RANDOM_STATE)

    # Preprocessing is happening same as data training using pipeline
    X_sample_transformed = preprocessor.transform(X_sample)

    # Instantiating SHAP explainer class and getting values from it
    explainer = shap. TreeExplainer(rf_model)
    shap_values = explainer(X_sample_transformed)

    return model, feat_imp, shap_values, X_sample_transformed, feature_names

def train_xgboost(df):
    """
    Train XGBoost model for data usage prediction.
    """

    target_col = "Data Usage (MB/day)"

    df_ml = df.copy()

    # One-hot encode any categorical columns
    df_ml = pd. get_dummies(df_ml, drop_first=True)

    X = df_ml.drop(columns=[target_col, 'User Behavior Class', "User ID"])
    y = df_ml[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("\n=== XGBoost Results ===")
    rmse_xgb = np. sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print("RMSE:", rmse_xgb)
    print("MAE:", mae_xgb)

    # 1. Access feature importances from the retrained XGBoost model
    feature_importances_updated = xgb_model.feature_importances_

    # 2. Create a Pandas DataFrame
    feature_names_updated = X_train.columns
    df_feature_importances_updated = pd.DataFrame({
        'Feature': feature_names_updated,
        'Importance': feature_importances_updated
    })

    # 3. Sort the DataFrame by importance scores in descending order
    df_feature_importances_updated = df_feature_importances_updated. sort_values(by='Importance', ascending=False)

    print("Feature Importances from Updated XGBoost Model (Sorted):")
    print(df_feature_importances_updated. head(10))

    return xgb_model, df_feature_importances_updated


if __name__ == "__main__":
    """
    Execute the regression analysis pipeline.
    """

    # Load data
    df = read_dataset()
    
    # Train Random Forest
    print("\n=== Random Forest - Battery Drain ===")
    rf_model, feat_imp_rf, shap_values_rf, X_sample_transformed_rf, feature_names_rf = train_random_forest(df)
    
    # Train XGBoost
    print("\n=== XGBoost - Data Usage ===")
    xgb_model, feat_imp_xgb = train_xgboost(df)
    
    # Prepare results for visualization
    results = {
        'rf': {
            'feature_importance': feat_imp_rf,
            'shap_values': shap_values_rf,
            'X_sample_transformed': X_sample_transformed_rf,
            'feature_names': feature_names_rf
        },
        'xgb': {
            'feature_importance': feat_imp_xgb
        }
    }
    
    # Generate and save all visualizations
    output_dir = PROJECT_ROOT / 'data'
    create_all_regression_visualizations(results, output_dir)
