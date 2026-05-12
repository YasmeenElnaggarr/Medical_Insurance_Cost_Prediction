import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics


# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)


# =========================
# App Title
# =========================
st.title("💰 Insurance Cost Prediction")
st.write("This app analyzes insurance data and predicts medical insurance charges using machine learning.")


# =========================
# Load Dataset
# =========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "insurance.csv"


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "insurance.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data


insurance_dataset = load_data()

# =========================
# Show Dataset
# =========================
st.header("📌 Dataset Overview")

st.subheader("First 5 Rows")
st.dataframe(insurance_dataset.head())

st.subheader("Dataset Shape")
st.write(f"Rows: {insurance_dataset.shape[0]}")
st.write(f"Columns: {insurance_dataset.shape[1]}")

st.subheader("Columns")
st.write(list(insurance_dataset.columns))

st.subheader("Missing Values")
st.dataframe(insurance_dataset.isnull().sum().reset_index().rename(
    columns={"index": "Column", 0: "Missing Values"}
))

st.subheader("Statistical Description")
st.dataframe(insurance_dataset.describe())


# =========================
# Data Preprocessing
# =========================
df = insurance_dataset.copy()

# Encoding categorical columns
df.replace({"sex": {"male": 0, "female": 1}}, inplace=True)
df.replace({"smoker": {"no": 0, "yes": 1}}, inplace=True)
df.replace({
    "region": {
        "southeast": 0,
        "southwest": 1,
        "northeast": 2,
        "northwest": 3
    }
}, inplace=True)


# =========================
# EDA
# =========================
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Charges Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(insurance_dataset["charges"], kde=True, ax=ax)
    ax.set_title("Distribution of Insurance Charges")
    st.pyplot(fig)

with col2:
    st.subheader("BMI Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(insurance_dataset["bmi"], kde=True, ax=ax)
    ax.set_title("Distribution of BMI")
    st.pyplot(fig)


col3, col4 = st.columns(2)

with col3:
    st.subheader("Charges by Smoker")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=insurance_dataset, x="smoker", y="charges", ax=ax)
    ax.set_title("Charges by Smoker")
    st.pyplot(fig)

with col4:
    st.subheader("Charges by Sex")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=insurance_dataset, x="sex", y="charges", ax=ax)
    ax.set_title("Charges by Sex")
    st.pyplot(fig)


st.subheader("Age vs Charges")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=insurance_dataset,
    x="age",
    y="charges",
    hue="smoker",
    ax=ax
)
ax.set_title("Age vs Charges")
st.pyplot(fig)


st.subheader("BMI vs Charges")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=insurance_dataset,
    x="bmi",
    y="charges",
    hue="smoker",
    ax=ax
)
ax.set_title("BMI vs Charges")
st.pyplot(fig)


st.subheader("Correlation Heatmap for Numerical Columns")
numerical_columns = insurance_dataset[["age", "bmi", "children", "charges"]]
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
    df.corr(),
    annot=True,
    cmap="rocket",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
ax.set_title("Full Correlation Heatmap")
st.pyplot(fig)


# =========================
# Split Data
# =========================
X = df.drop(columns="charges", axis=1)
Y = df["charges"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)


# =========================
# Train Models
# =========================
linear_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()

linear_model.fit(X_train, Y_train)
lasso_model.fit(X_train, Y_train)
ridge_model.fit(X_train, Y_train)


# =========================
# Evaluation Function
# =========================
def evaluate_model(model, model_name):
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_r2 = metrics.r2_score(Y_train, train_prediction)
    test_r2 = metrics.r2_score(Y_test, test_prediction)
    mae = metrics.mean_absolute_error(Y_test, test_prediction)
    mse = metrics.mean_squared_error(Y_test, test_prediction)
    rmse = np.sqrt(mse)

    return {
        "Model": model_name,
        "Training R2 Score": train_r2,
        "Testing R2 Score": test_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


results = [
    evaluate_model(linear_model, "Linear Regression"),
    evaluate_model(lasso_model, "Lasso Regression"),
    evaluate_model(ridge_model, "Ridge Regression")
]

results_df = pd.DataFrame(results)


# =========================
# Model Results
# =========================
st.header("🤖 Model Training Results")

st.subheader("Model Comparison")
st.dataframe(results_df)

best_model_row = results_df.sort_values(by="Testing R2 Score", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]

st.success(f"Best Model: {best_model_name}")

if best_model_name == "Linear Regression":
    best_model = linear_model
elif best_model_name == "Lasso Regression":
    best_model = lasso_model
else:
    best_model = ridge_model


# =========================
# True vs Predicted
# =========================
st.subheader("True vs Predicted Values")

test_prediction = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Y_test, test_prediction)
ax.set_xlabel("True Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"True vs Predicted - {best_model_name}")
st.pyplot(fig)


# =========================
# Coefficients
# =========================
st.subheader("Model Coefficients")

coefficients_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": best_model.coef_
})

st.write("Intercept:", best_model.intercept_)
st.dataframe(coefficients_df)


# =========================
# Prediction Section
# =========================
st.header("🔮 Predict Insurance Cost")

st.write("Enter customer information to predict insurance charges.")

col_a, col_b = st.columns(2)

with col_a:
    age = st.number_input("Age", min_value=18, max_value=100, value=31)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.74)
    
with col_b:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox(
        "Region",
        ["southeast", "southwest", "northeast", "northwest"]
    )


# Encoding user input
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
st.dataframe(input_data)


if st.button("Predict Insurance Cost"):
    prediction = best_model.predict(input_data)

    st.success(f"The estimated insurance cost is: ${prediction[0]:,.2f}")


# =========================
# Final Insights
# =========================
st.header("📌 Final Insights")

average_smoker_charges = insurance_dataset.groupby("smoker")["charges"].mean()
average_region_charges = insurance_dataset.groupby("region")["charges"].mean()

st.write("Average charges by smoker status:")
st.dataframe(average_smoker_charges)

st.write("Average charges by region:")
st.dataframe(average_region_charges)

st.info("This project is for educational purposes and demonstrates data analysis and machine learning using Python and Streamlit.")