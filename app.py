import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    if os.path.exists("data/customer_segments.csv"):
        df = pd.read_csv("data/customer_segments.csv")

    elif os.path.exists("data/Mall_Customers.csv"):
        df = pd.read_csv("data/Mall_Customers.csv")

    else:
        st.error(
            "Dataset not found. Please make sure "
            "data/customer_segments.csv is available in the repository."
        )
        st.stop()

    df.columns = df.columns.str.strip()

    return df


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_model(df):

    X = df[FEATURES].copy()

    # Preprocessing
    preprocessor = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    X_scaled = preprocessor.fit_transform(X)

    # Find best K using Silhouette Score
    k_values = range(2, 11)

    silhouette_scores = []
    inertias = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)

        silhouette_scores.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )

    # Select K with highest silhouette score
    best_k = list(k_values)[
        np.argmax(silhouette_scores)
    ]

    # Final model
    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50
    )

    clusters = model.fit_predict(X_scaled)

    # Copy dataframe
    result_df = df.copy()

    result_df["Cluster"] = clusters

    final_score = silhouette_score(
        X_scaled,
        clusters
    )

    return (
        preprocessor,
        model,
        result_df,
        best_k,
        final_score,
        list(k_values),
        inertias,
        silhouette_scores
    )


# ---------------------------------------------------------
# LOAD DATA AND MODEL
# ---------------------------------------------------------

df = load_data()

(
    preprocessor,
    model,
    segmented_df,
    best_k,
    silhouette,
    k_values,
    inertias,
    silhouette_scores
) = train_model(df)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("👥 Mall Customer Segmentation")

st.caption(
    "K-Means clustering using the Kaggle Mall Customer "
    "Segmentation Dataset"
)


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

info1, info2 = st.columns(2)

with info1:
    st.metric(
        "Best K",
        best_k
    )

with info2:
    st.metric(
        "Silhouette Score",
        f"{silhouette:.4f}"
    )


# ---------------------------------------------------------
# CUSTOMER SEGMENT VISUALIZATION
# ---------------------------------------------------------

st.subheader("📊 Customer Segments")

fig = px.scatter(
    segmented_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=segmented_df["Cluster"].astype(str),
    hover_data=[
        "CustomerID",
        "Gender",
        "Age"
    ],
    labels={
        "color": "Cluster"
    },
    title="Customer Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

st.subheader("📋 Cluster Profiles")

profile = (
    segmented_df
    .groupby("Cluster")[FEATURES]
    .mean()
    .round(2)
)

st.dataframe(
    profile,
    use_container_width=True
)


# ---------------------------------------------------------
# ELBOW AND SILHOUETTE DATA
# ---------------------------------------------------------

with st.expander("📈 Model Evaluation"):

    evaluation_df = pd.DataFrame({
        "K": list(k_values),
        "Inertia": inertias,
        "Silhouette Score": silhouette_scores
    })

    st.dataframe(
        evaluation_df.round(4),
        use_container_width=True
    )


# ---------------------------------------------------------
# NEW CUSTOMER PREDICTION
# ---------------------------------------------------------

st.divider()

st.subheader("🔮 Predict Customer Segment")

st.write(
    "Enter the details of a new customer to predict "
    "their K-Means customer segment."
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
        "Predict Segment",
        type="primary"
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if submitted:

    new_customer = pd.DataFrame({
        "Age": [float(age)],
        "Annual Income (k$)": [float(income)],
        "Spending Score (1-100)": [float(spending)]
    })

    # Apply the SAME preprocessing fitted during model training
    new_customer_scaled = preprocessor.transform(
        new_customer[FEATURES]
    )

    # Predict cluster
    predicted_cluster = int(
        model.predict(new_customer_scaled)[0]
    )

    st.success(
        f"🎯 Predicted Customer Segment: Cluster {predicted_cluster}"
    )

    # Show profile of predicted cluster
    predicted_profile = profile.loc[
        predicted_cluster
    ]

    st.subheader("Predicted Cluster Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    with profile_col1:
        st.metric(
            "Average Age",
            f"{predicted_profile['Age']:.1f}"
        )

    with profile_col2:
        st.metric(
            "Average Income",
            f"${predicted_profile['Annual Income (k$)']:.1f}k"
        )

    with profile_col3:
        st.metric(
            "Average Spending",
            f"{predicted_profile['Spending Score (1-100)']:.1f}"
        )

    st.info(
        "Cluster numbers are model-generated labels. "
        "Use the cluster profile to understand the business meaning "
        "of the segment."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Developed using Python, Scikit-learn, K-Means and Streamlit."
)