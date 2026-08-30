# Mall Customer Segmentation using K-Means

A complete ML project based on the **Kaggle Mall Customer Segmentation Dataset**.

**Kaggle dataset:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

The dataset contains 200 customers and five columns: CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1–100). The model uses Age, Annual Income and Spending Score as clustering features; CustomerID is an identifier and Gender is retained for analysis.

## Project structure
```text
customer-segmentation-kaggle/
├── data/
│   ├── Mall_Customers.csv          # download from Kaggle
│   └── customer_segments.csv       # generated after training
├── models/
│   ├── kmeans_model.pkl
│   ├── preprocessor.pkl
│   ├── features.pkl
│   ├── k_selection_metrics.csv
│   ├── cluster_profile.csv
│   ├── elbow_plot.png
│   └── silhouette_plot.png
├── notebooks/
│   └── customer_segmentation.ipynb
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
Install:
```bash
pip install -r requirements.txt
```

## Dataset
Download `Mall_Customers.csv` from Kaggle and put it inside `data/`.

## Train
```bash
python train_model.py
```

This generates the trained K-Means model, preprocessing pipeline, cluster assignments, profiles and evaluation plots.

## Run the UI
```bash
streamlit run app.py
```

## Evaluation
The project uses:
- Elbow Method / inertia
- Silhouette Score

The training script tests K=2 through K=10 and automatically selects the K with the highest silhouette score.

## Model improvement
Discuss scaling, comparison of multiple K values, silhouette analysis, feature-set comparison and possible comparison with DBSCAN/Hierarchical Clustering.

## GitHub
```bash
git init
git add .
git commit -m "Build Kaggle mall customer segmentation"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Deployment
Deploy `app.py` using Streamlit Community Cloud. Keep `requirements.txt` and the generated `models/` files in the repository.

**Dataset attribution:** Mall Customer Segmentation Dataset, Kaggle, vjchoudhary7. Check the current Kaggle license/terms before redistributing the CSV.
