import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. DEFINE CUSTOM TRANSFORMER (Required by joblib to load the pipeline)
# ==============================================================================
class SelectiveOutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5, target_columns=None):
        self.factor = factor
        self.target_columns = target_columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (lower_limit, upper_limit) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = np.clip(X_df[col], lower_limit, upper_limit)
        return X_df.values

# ==============================================================================
# 2. LOAD MODEL PIPELINE
# ==============================================================================
@st.cache_resource
def load_pipeline():
    model_path = Path.cwd() / "house_price_champion_pipeline.pkl"
    return joblib.load(model_path)

model = load_pipeline()

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 House Price Prediction")
st.write("Enter the property details below to estimate the market value.")

st.subheader("Property Details")

col1, col2 = st.subplots(2)

with col1:
    MSSubClass = st.selectbox(
        "MS SubClass (Dwelling Type)",
        ["20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "160", "180", "190"],
        index=5 # default to 60
    )
    LotArea = st.number_input("Lot Area (sq ft)", min_value=100, max_value=300000, value=8500, step=100)
    OverallCond = st.slider("Overall Condition (1-10)", min_value=1, max_value=10, value=5)
    YearBuilt = st.number_input("Year Built", min_value=1850, max_value=2026, value=2000)
    YearRemodAdd = st.number_input("Year Remodeled / Added", min_value=1850, max_value=2026, value=2005)

with col2:
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0.0, value=850.0, step=25.0)
    BsmtFinSF2 = st.number_input("Finished Basement 2 Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
    
    MSZoning = st.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
    LotConfig = st.selectbox("Lot Configuration", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])
    BldgType = st.selectbox("Building Type", ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"])
    Exterior1st = st.selectbox(
        "Exterior Covering 1st",
        ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "Plywood", "CemntBd", 
         "BrkFace", "WdShing", "Stucco", "AsbShng", "Stone", "ImStucc", "CBlock", "BrkComm", "AsphShn"]
    )

# ==============================================================================
# 4. PREDICTION LOGIC WITH MATCHING FEATURE ENGINEERING
# ==============================================================================
if st.button("🚀 Predict House Price", use_container_width=True):
    
    # 1. Compute Engineered Features (Matching training pipeline exactly)
    has_bsmt_fin2 = int(BsmtFinSF2 > 0)
    lot_area_log = np.log1p(LotArea)
    house_age = 2026 - YearBuilt
    years_since_remodel = 2026 - YearRemodAdd
    is_remodeled = int(YearRemodAdd != YearBuilt)

    # 2. Assemble DataFrame with exact columns expected by ColumnTransformer
    input_df = pd.DataFrame([{
        "MSSubClass": str(MSSubClass),
        "MSZoning": MSZoning,
        "LotArea": lot_area_log,
        "LotConfig": LotConfig,
        "BldgType": BldgType,
        "OverallCond": OverallCond,
        "YearRemodAdd": YearRemodAdd,
        "Exterior1st": Exterior1st,
        "BsmtFinSF2": BsmtFinSF2,
        "TotalBsmtSF": TotalBsmtSF,
        "Has_BsmtFin2": has_bsmt_fin2,
        "HouseAge": house_age,
        "YearsSinceRemodel": years_since_remodel,
        "IsRemodeled": is_remodeled
    }])

    # 3. Predict Log Price and Inverse-Transform to Dollars
    predicted_log_price = model.predict(input_df)[0]
    predicted_dollars = np.expm1(predicted_log_price)

    # 4. Display Result
    st.success(f"### 🏷️ Estimated House Price: **${predicted_dollars:,.2f}**")

