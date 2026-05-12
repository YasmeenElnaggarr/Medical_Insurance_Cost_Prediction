from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics


# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)


# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 18px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.05);
    }

    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3rem;
        font-size: 18px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# Header
# ==============================
st.markdown(
    '<p class="main-title">💰 Medical Insurance Cost Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">A Streamlit machine learning app for analyzing and predicting medical insurance charges.</p>',
    unsafe_allow_html=True
)

st.divider()


# ==============================
# Dataset Loader
# ==============================
BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_DATA_PATHS = [
    BASE_DIR / "insurance.csv",
    BASE_DIR / "data" / "insurance.csv",
    BASE_DIR.parent / "insurance.csv"
]


@st.cache_data
def load_data_from_path(path):
    return pd.read_csv(path)


@st.cache_data
def load_data_from_uploaded_file(uploaded_file):
    return pd.read_csv(uploaded_file)


st.sidebar.title("⚙️ App Controls")
uploaded_file = st.sidebar.file_uploader("Upload insurance.csv", type=["csv"])

insurance_dataset = None
used_path = None

if uploaded_file is not None:
    insurance_dataset = load_data_from_uploaded_file(uploaded_file)
    used_path = "Uploaded File"
else:
    for path in POSSIBLE_DATA_PATHS:
        if path.exists():
            insurance_dataset = load_data_from_path(path)
            used_path = str(path)
            break

if insurance_dataset is None:
    st.error("insurance.csv file was not found.")
    st.write("Put `insurance.csv` in the same folder as `Medical.py` on GitHub.")
    st.write("Current project folder:")
    st.code(str(BASE_DIR))

    st.write("Files found in current folder:")
    try:
        st.write([file.name for file in BASE_DIR.iterdir()])
    except Exception:
        st.write("Could not list files.")

    st.stop()


# ==============================
# Dataset Validation
# ==============================
required_columns = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges"
]

missing_columns = [
    col for col in required_columns
    if col not in insurance_dataset.columns
]

if missing_columns:
    st.error(f"The dataset is missing these columns: {missing_columns}")
    st.stop()


# ==============================
# Clean Dataset
# ==============================
df_original = insurance_dataset.copy()

df_original["sex"] = df_original["sex"].astype(str).str.lower().str.strip()
df_original["smoker"] = df_original["smoker"].astype(str).str.lower().str.strip()
df_original["region"] = df_original["region"].astype(str).str.lower().str.strip()


def preprocess_data(data):
    df = data.copy()

    df["sex"] = df["sex"].map({
        "male": 0,
        "female": 1
    })

    df["smoker"] = df["smoker"].map({
        "no": 0,
        "yes": 1
    })

    df["region"] = df["region"].map({
        "southeast": 0,
        "southwest": 1,
        "northeast": 2,
        "northwest": 3
    })

    return df


df_encoded = preprocess_data(df_original)

if df_encoded.isnull().sum().sum() > 0:
    st.warning("Some values could not be encoded. Please check sex, smoker, or region values.")
    st.dataframe(df_encoded[df_encoded.isnull().any(axis=1)], use_container_width=True)
    st.stop()


# ==============================
# Sidebar Model Settings
# ==============================
st.sidebar.subheader("🤖 Model Settings")

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)

alpha_value = st.sidebar.slider(
    "Alpha for Lasso and Ridge",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01
)


# ==============================
# Split Data
# ==============================
X = df_encoded.drop(columns=["charges"])
Y = df_encoded["charges"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=test_size,
    random_state=random_state
)


# ==============================
# Train Models
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=alpha_value, max_iter=10000),
    "Ridge Regression": Ridge(alpha=alpha_value)
}

trained_models = {}
results = []

for model_name, model in models.items():
    model.fit(X_train, Y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = metrics.r2_score(Y_train, train_predictions)
    test_r2 = metrics.r2_score(Y_test, test_predictions)
    mae = metrics.mean_absolute_error(Y_test, test_predictions)
    mse = metrics.mean_squared_error(Y_test, test_predictions)
    rmse = np.sqrt(mse)

    trained_models[model_name] = model

    results.append({
        "Model": model_name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

results_df = pd.DataFrame(results)

best_model_row = results_df.sort_values(
    by="Test R2",
    ascending=False
).iloc[0]

best_model_name = best_model_row["Model"]
best_model = trained_models[best_model_name]


# ==============================
# Main Metrics
# ==============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", df_original.shape[0])

with col2:
    st.metric("Columns", df_original.shape[1])

with col3:
    st.metric("Best Model", best_model_name)

with col4:
    st.metric("Best Test R2", f"{best_model_row['Test R2']:.3f}")

st.caption(f"Dataset source: {used_path}")


# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Dataset Overview",
    "📊 Data Analysis",
    "🤖 Model Results",
    "🔮 Prediction",
    "📎 Final Insights"
])


# ==============================
# Helper Function for Counts
# ==============================
def get_value_counts(data, column_name):
    counts_df = (
        data[column_name]
        .value_counts()
        .rename_axis(column_name)
        .reset_index(name="count")
    )
    return counts_df


# ==============================
# Tab 1: Dataset Overview
# ==============================
with tab1:
    st.markdown(
        '<p class="section-title">Dataset Preview</p>',
        unsafe_allow_html=True
    )

    st.dataframe(df_original.head(20), use_container_width=True)

    st.markdown(
        '<p class="section-title">Dataset Information</p>',
        unsafe_allow_html=True
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("Dataset Shape:")
        st.write(f"Rows: **{df_original.shape[0]}**")
        st.write(f"Columns: **{df_original.shape[1]}**")

    with info_col2:
        st.write("Column Names:")
        st.write(list(df_original.columns))

    st.markdown(
        '<p class="section-title">Missing Values</p>',
        unsafe_allow_html=True
    )

    missing_values = df_original.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, use_container_width=True)

    st.markdown(
        '<p class="section-title">Statistical Summary</p>',
        unsafe_allow_html=True
    )

    st.dataframe(df_original.describe(), use_container_width=True)

    st.markdown(
        '<p class="section-title">Categorical Columns Summary</p>',
        unsafe_allow_html=True
    )

    cat_col1, cat_col2, cat_col3 = st.columns(3)

    with cat_col1:
        st.write("Sex Values")
        sex_counts = get_value_counts(df_original, "sex")
        st.dataframe(sex_counts, use_container_width=True)

    with cat_col2:
        st.write("Smoker Values")
        smoker_counts = get_value_counts(df_original, "smoker")
        st.dataframe(smoker_counts, use_container_width=True)

    with cat_col3:
        st.write("Region Values")
        region_counts = get_value_counts(df_original, "region")
        st.dataframe(region_counts, use_container_width=True)


# ==============================
# Tab 2: Data Analysis
# ==============================
with tab2:
    st.markdown(
        '<p class="section-title">Exploratory Data Analysis</p>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Charges Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_original["charges"], kde=True, ax=ax)
        ax.set_title("Distribution of Insurance Charges")
        ax.set_xlabel("Charges")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with chart_col2:
        st.subheader("BMI Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_original["bmi"], kde=True, ax=ax)
        ax.set_title("Distribution of BMI")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Charges by Smoker")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_original, x="smoker", y="charges", ax=ax)
        ax.set_title("Charges by Smoker")
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Charges")
        st.pyplot(fig)

    with chart_col4:
        st.subheader("Charges by Sex")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_original, x="sex", y="charges", ax=ax)
        ax.set_title("Charges by Sex")
        ax.set_xlabel("Sex")
        ax.set_ylabel("Charges")
        st.pyplot(fig)

    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        st.subheader("Average Charges by Region")
        region_avg = df_original.groupby("region")["charges"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=region_avg, x="region", y="charges", ax=ax)
        ax.set_title("Average Charges by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Average Charges")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    with chart_col6:
        st.subheader("Average Charges by Children")
        children_avg = df_original.groupby("children")["charges"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=children_avg, x="children", y="charges", ax=ax)
        ax.set_title("Average Charges by Number of Children")
        ax.set_xlabel("Children")
        ax.set_ylabel("Average Charges")
        st.pyplot(fig)

    st.subheader("Age vs Charges")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_original,
        x="age",
        y="charges",
        hue="smoker",
        ax=ax
    )
    ax.set_title("Age vs Charges")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

    st.subheader("BMI vs Charges")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_original,
        x="bmi",
        y="charges",
        hue="smoker",
        ax=ax
    )
    ax.set_title("BMI vs Charges")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap for Numerical Columns")

    numerical_columns = df_original[[
        "age",
        "bmi",
        "children",
        "charges"
    ]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        numerical_columns.corr(),
        annot=True,
        cmap="rocket",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Heatmap for Numerical Features")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap After Encoding")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df_encoded.corr(),
        annot=True,
        cmap="rocket",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Full Correlation Heatmap")
    st.pyplot(fig)


# ==============================
# Tab 3: Model Results
# ==============================
with tab3:
    st.markdown(
        '<p class="section-title">Model Training Results</p>',
        unsafe_allow_html=True
    )

    display_results = results_df.copy()
    display_results["Train R2"] = display_results["Train R2"].round(4)
    display_results["Test R2"] = display_results["Test R2"].round(4)
    display_results["MAE"] = display_results["MAE"].round(2)
    display_results["MSE"] = display_results["MSE"].round(2)
    display_results["RMSE"] = display_results["RMSE"].round(2)

    st.dataframe(display_results, use_container_width=True)

    st.success(f"Best Model Based on Test R2 Score: {best_model_name}")

    selected_model_name = st.selectbox(
        "Choose a model to inspect:",
        list(trained_models.keys())
    )

    selected_model = trained_models[selected_model_name]

    train_pred = selected_model.predict(X_train)
    test_pred = selected_model.predict(X_test)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Train R2", f"{metrics.r2_score(Y_train, train_pred):.4f}")

    with metric_col2:
        st.metric("Test R2", f"{metrics.r2_score(Y_test, test_pred):.4f}")

    with metric_col3:
        st.metric("MAE", f"{metrics.mean_absolute_error(Y_test, test_pred):.2f}")

    with metric_col4:
        rmse_value = np.sqrt(metrics.mean_squared_error(Y_test, test_pred))
        st.metric("RMSE", f"{rmse_value:.2f}")

    st.subheader("True vs Predicted Values")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(Y_train, train_pred, marker="o", label="Train Predictions")
    ax.scatter(Y_test, test_pred, marker="^", label="Test Predictions")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"True vs Predicted Values - {selected_model_name}")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Model Coefficients")

    coefficients_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": selected_model.coef_
    })

    st.write(f"Intercept: **{selected_model.intercept_:.4f}**")
    st.dataframe(coefficients_df, use_container_width=True)


# ==============================
# Tab 4: Prediction
# ==============================
with tab4:
    st.markdown(
        '<p class="section-title">Predict Insurance Cost</p>',
        unsafe_allow_html=True
    )

    prediction_model_name = st.selectbox(
        "Choose model for prediction:",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )

    prediction_model = trained_models[prediction_model_name]

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=31,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.74,
            step=0.1
        )

    with input_col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        smoker = st.selectbox(
            "Smoker",
            ["no", "yes"]
        )

        region = st.selectbox(
            "Region",
            ["southeast", "southwest", "northeast", "northwest"]
        )

    sex_encoded = 0 if sex == "male" else 1
    smoker_encoded = 0 if smoker == "no" else 1

    region_mapping = {
        "southeast": 0,
        "southwest": 1,
        "northeast": 2,
        "northwest": 3
    }

    region_encoded = region_mapping[region]

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_encoded,
        "region": region_encoded
    }])

    st.subheader("Input Data After Encoding")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Predict Insurance Cost"):
        prediction = prediction_model.predict(input_data)[0]

        st.success(f"The estimated insurance cost is: ${prediction:,.2f}")

        st.metric(
            label="Predicted Insurance Charges",
            value=f"${prediction:,.2f}"
        )


# ==============================
# Tab 5: Final Insights
# ==============================
with tab5:
    st.markdown(
        '<p class="section-title">Final Insights</p>',
        unsafe_allow_html=True
    )

    smoker_avg = (
        df_original
        .groupby("smoker")["charges"]
        .mean()
        .sort_values(ascending=False)
    )

    region_avg = (
        df_original
        .groupby("region")["charges"]
        .mean()
        .sort_values(ascending=False)
    )

    sex_avg = (
        df_original
        .groupby("sex")["charges"]
        .mean()
        .sort_values(ascending=False)
    )

    st.subheader("Average Charges by Smoker Status")
    smoker_avg_df = smoker_avg.reset_index()
    smoker_avg_df.columns = ["smoker", "Average Charges"]
    st.dataframe(smoker_avg_df, use_container_width=True)

    st.subheader("Average Charges by Region")
    region_avg_df = region_avg.reset_index()
    region_avg_df.columns = ["region", "Average Charges"]
    st.dataframe(region_avg_df, use_container_width=True)

    st.subheader("Average Charges by Sex")
    sex_avg_df = sex_avg.reset_index()
    sex_avg_df.columns = ["sex", "Average Charges"]
    st.dataframe(sex_avg_df, use_container_width=True)

    st.write("### Main Findings")

    highest_smoker = smoker_avg.index[0]
    highest_region = region_avg.index[0]
    highest_sex = sex_avg.index[0]

    st.write(f"- The highest average charges based on smoking status are for: **{highest_smoker}**")
    st.write(f"- The highest average charges based on region are in: **{highest_region}**")
    st.write(f"- The highest average charges based on sex are for: **{highest_sex}**")
    st.write(f"- The best model in this run is: **{best_model_name}**")
    st.write(f"- Best Test R2 Score: **{best_model_row['Test R2']:.4f}**")

    st.info(
        "This project is for educational purposes. Real insurance pricing may require more complex medical, actuarial, and financial factors."
    )


# ==============================
# Footer
# ==============================
st.divider()
st.caption("Created as a Machine Learning Streamlit Project.")
