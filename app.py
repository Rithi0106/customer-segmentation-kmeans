import os, joblib, pandas as pd
import streamlit as st
import plotly.express as px

FEATURES=["Age","Annual Income (k$)","Spending Score (1-100)"]
st.set_page_config(page_title="Mall Customer Segmentation",page_icon="👥",layout="wide")

@st.cache_resource
def load():
    return joblib.load("models/kmeans_model.pkl"),joblib.load("models/preprocessor.pkl")
@st.cache_data
def data():
    p="data/customer_segments.csv"
    return pd.read_csv(p if os.path.exists(p) else "data/Mall_Customers.csv")

model,prep=load(); df=data()
st.title("👥 Mall Customer Segmentation")
st.caption("K-Means clustering using the Kaggle Mall Customer Segmentation Dataset")

a,b,c,d=st.columns(4)
a.metric("Customers",len(df))
b.metric("Clusters",int(df["Cluster"].nunique()) if "Cluster" in df else model.n_clusters)
c.metric("Avg Income",f"${df['Annual Income (k$)'].mean():.1f}k")
d.metric("Avg Spending",f"{df['Spending Score (1-100)'].mean():.1f}")

if "Cluster" in df:
    fig=px.scatter(df,x="Annual Income (k$)",y="Spending Score (1-100)",color=df["Cluster"].astype(str),hover_data=["CustomerID","Gender","Age"],labels={"color":"Cluster"},title="Customer Segments")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Cluster Profiles")
    st.dataframe(df.groupby("Cluster")[FEATURES].mean().round(2),use_container_width=True)

st.divider(); st.subheader("🔮 Predict New Customer Segment")
with st.form("prediction"):
    x,y,z=st.columns(3)
    age=x.number_input("Age",18,80,30)
    income=y.number_input("Annual Income (k$)",1,300,60)
    spending=z.slider("Spending Score",1,100,50)
    submit=st.form_submit_button("Predict Segment",type="primary")
if submit:
    inp=pd.DataFrame([{"Age":age,"Annual Income (k$)":income,"Spending Score (1-100)":spending}])[FEATURES]
    cluster=int(model.predict(prep.transform(inp))[0])
    st.success(f"Predicted Cluster: {cluster}")
    st.info("Cluster numbers are model-generated labels. Interpret them using the cluster profile shown above.")
