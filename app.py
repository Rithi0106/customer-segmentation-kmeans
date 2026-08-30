import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    possible_files = [
        "data/customer_segments.csv",
        "data/Mall_Customers.csv",
        "customer_segments.csv",
        "Mall_Customers.csv"
    ]

    for file in possible_files:

        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            return df

    st.error(
        "Dataset not found. Please make sure the CSV file "
        "is available in the repository."
    )

    st.stop()


# =========================================================
# TRAIN K-MEANS
# =========================================================

@st.cache_resource
def train_model(df):

    # Select clustering features
    X = df[FEATURES].copy()

    # Convert values to numeric
    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Fill missing values manually
    # This avoids SimpleImputer compatibility problems
    for column in FEATURES:
        X[column] = X[column].fillna(
            X[column].median()
        )

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Test different K values
    k_values = range(2, 11)

    inertias = []
    silhouette_scores = []

    for k in k_values:

        temp_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = temp_model.fit_predict(X_scaled)

        inertias.append(
            temp_model.inertia_
        )

        silhouette_scores.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )

    # Select best K using silhouette score
    best_index = int(
        np.argmax(silhouette_scores)
    )

    best_k = list(k_values)[best_index]

    # Final K-Means model
    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50
    )

    clusters = model.fit_predict(X_scaled)

    # Add clusters
    result_df = df.copy()

    result_df["Cluster"] = clusters

    final_score = silhouette_score(
        X_scaled,
        clusters
    )

    return (
        scaler,
        model,
        result_df,
        best_k,
        final_score,
        list(k_values),
        inertias,
        silhouette_scores
    )


# =========================================================
# LOAD DATA + TRAIN
# =========================================================

df = load_data()

(
    scaler,
    model,
    segmented_df,
    best_k,
    silhouette,
    k_values,
    inertias,
    silhouette_scores
) = train_model(df)


# =========================================================
# TITLE
# =========================================================

st.title("👥 Mall Customer Segmentation")

st.write(
    "Customer segmentation using K-Means clustering "
    "on the Kaggle Mall Customer Segmentation Dataset."
)


# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Customers",
        len(segmented_df)
    )

with col2:
    st.metric(
        "Clusters",
        best_k
    )

with col3:
    st.metric(
        "Avg Income",
        f"${segmented_df['Annual Income (k$)'].mean():.1f}k"
    )

with col4:
    st.metric(
        "Avg Spending",
        f"{segmented_df['Spending Score (1-100)'].mean():.1f}"
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best K",
        best_k
    )

with col2:
    st.metric(
        "Silhouette Score",
        f"{silhouette:.4f}"
    )


# =========================================================
# CUSTOMER SEGMENT VISUALIZATION
# =========================================================

st.subheader("📊 Customer Segments")

plot_df = segmented_df.copy()

plot_df["Cluster"] = plot_df["Cluster"].astype(str)

fig = px.scatter(
    plot_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    hover_data=[
        "CustomerID",
        "Gender",
        "Age"
    ],
    title="Customer Segmentation using K-Means"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =========================================================
# CLUSTER PROFILE
# =========================================================

st.subheader("📋 Cluster Profiles")

profile = (
    segmented_df
    .groupby("Cluster")[FEATURES]
    .mean()
    .round(2)
)

st.dataframe(
    profile,
    width="stretch"
)


# =========================================================
# MODEL EVALUATION TABLE
# =========================================================

with st.expander("📈 K Selection Results"):

    evaluation_df = pd.DataFrame({
        "K": list(k_values),
        "Inertia": inertias,
        "Silhouette Score": silhouette_scores
    })

    st.dataframe(
        evaluation_df.round(4),
        width="stretch"
    )


# =========================================================
# NEW CUSTOMER PREDICTION
# =========================================================

st.divider()

st.subheader("🔮 Predict Customer Segment")

st.write(
    "Enter the details of a new customer."
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        income = st.number_input(
            "Annual Income (k$)",
            min_value=1,
            max_value=300,
            value=60
        )

    with col3:
        spending = st.slider(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50
        )

    submitted = st.form_submit_button(
        "Predict Segment"
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    new_customer = pd.DataFrame({
        "Age": [float(age)],
        "Annual Income (k$)": [float(income)],
        "Spending Score (1-100)": [float(spending)]
    })

    # Scale new customer using the scaler
    new_customer_scaled = scaler.transform(
        new_customer[FEATURES]
    )

    # Predict cluster
    predicted_cluster = int(
        model.predict(new_customer_scaled)[0]
    )

    st.success(
        f"🎯 Predicted Customer Segment: "
        f"Cluster {predicted_cluster}"
    )

    # Display cluster profile
    predicted_profile = profile.loc[
        predicted_cluster
    ]

    st.subheader(
        "📌 Predicted Cluster Profile"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Age",
            f"{predicted_profile['Age']:.1f}"
        )

    with col2:
        st.metric(
            "Average Income",
            f"${predicted_profile['Annual Income (k$)']:.1f}k"
        )

    with col3:
        st.metric(
            "Average Spending",
            f"{predicted_profile['Spending Score (1-100)']:.1f}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Mall Customer Segmentation | "
    "K-Means Clustering | "
    "Machine Learning Project"
)