import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import chardet

st.set_page_config(page_title="Analyse Client", layout="wide")
st.title("Application d'analyse e-commerce")

# Sidebar
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        raw_data = file.read(10000)
        encoding = chardet.detect(raw_data)['encoding']
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
            encoding = 'latin1'
        return df, encoding
    else:
        df = pd.read_excel(file)
        return df, 'Excel'

def show_descriptive_stats(df):
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    st.write("**Distribution des variables numériques :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

def apply_fp_growth(df):
    st.subheader("Analyse par FP-Growth")
    # Supposons que les colonnes soient 'InvoiceNo' et 'Description'
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    st.write("Règles générées :")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

def apply_kmeans(df):
    st.subheader("Segmentation par K-means")
    k = st.slider("Choisissez le nombre de clusters", 2, 10, 4)
    features = df.select_dtypes(include='number')
    X_scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)
    st.write("Extrait des clusters :")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='Set2')
    st.pyplot(fig)

def apply_rfm(df):
    st.subheader("Segmentation RFM")
    rfm = df[['Recence', 'Frequence', 'Montant']].copy()
    rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1,2,3,4])
    rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1,2,3,4])
    rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)
    st.dataframe(rfm.head())
    fig, ax = plt.subplots()
    sns.histplot(rfm['RFM_Score'], bins=10, kde=True)
    st.pyplot(fig)

if uploaded_file:
    df, encoding = load_data(uploaded_file)
st.success(f"Fichier chargé avec succès ✅ (encodage détecté : {encoding})")
st.write("Aperçu des données :")
st.dataframe(df.head())

    model_choice = st.sidebar.selectbox("Choisir une analyse", ["Statistiques descriptives", "FP-Growth", "K-means", "RFM"])

    if model_choice == "Statistiques descriptives":
        show_descriptive_stats(df)
    elif model_choice == "FP-Growth":
        apply_fp_growth(df)
    elif model_choice == "K-means":
        apply_kmeans(df)
    elif model_choice == "RFM":
        apply_rfm(df)
else:
    st.info("Veuillez charger un fichier pour commencer.")
