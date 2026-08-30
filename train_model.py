import os, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA="data/Mall_Customers.csv"
FEATURES=["Age","Annual Income (k$)","Spending Score (1-100)"]

df=pd.read_csv(DATA)
df.columns=df.columns.str.strip()
X=df[FEATURES]

prep=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
Xs=prep.fit_transform(X)

ks=range(2,11); inertia=[]; sil=[]
for k in ks:
    m=KMeans(n_clusters=k,random_state=42,n_init=20)
    y=m.fit_predict(Xs); inertia.append(m.inertia_); sil.append(silhouette_score(Xs,y))

best_k=list(ks)[int(np.argmax(sil))]
model=KMeans(n_clusters=best_k,random_state=42,n_init=50)
df["Cluster"]=model.fit_predict(Xs)
score=silhouette_score(Xs,df["Cluster"])

os.makedirs("models",exist_ok=True)
os.makedirs("data",exist_ok=True)
df.to_csv("data/customer_segments.csv",index=False)
joblib.dump(prep,"models/preprocessor.pkl")
joblib.dump(model,"models/kmeans_model.pkl")
joblib.dump(FEATURES,"models/features.pkl")
pd.DataFrame({"K":list(ks),"Inertia":inertia,"Silhouette_Score":sil}).to_csv("models/k_selection_metrics.csv",index=False)
df.groupby("Cluster")[FEATURES].mean().round(2).to_csv("models/cluster_profile.csv")

plt.figure(figsize=(8,5)); plt.plot(list(ks),inertia,marker="o"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Method"); plt.tight_layout(); plt.savefig("models/elbow_plot.png"); plt.close()
plt.figure(figsize=(8,5)); plt.plot(list(ks),sil,marker="o"); plt.xlabel("K"); plt.ylabel("Silhouette Score"); plt.title("Silhouette Analysis"); plt.tight_layout(); plt.savefig("models/silhouette_plot.png"); plt.close()

print("Best K:",best_k)
print("Silhouette Score:",round(score,4))
print(df.groupby("Cluster")[FEATURES].mean().round(2))
