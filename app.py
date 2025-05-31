import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import chardet

st.set_page_config(page_title="Analyse Client - E-commerce", layout="wide")
st.title("Application d'analyse e-commerce")

# --- Sidebar ---
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("Choix de l'analyse")
model_choice = st.sidebar.selectbox("Modèle d'analyse", ["Statistiques descriptives", "FP-Growth", "K-means", "RFM"])

@st.cache_data(show_spinner=False)
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
    st.write("**Moyenne et Variance des variables numériques :**")
    stats = pd.DataFrame({
        'Moyenne': df.select_dtypes(include='number').mean(),
        'Variance': df.select_dtypes(include='number').var()
    })
    st.dataframe(stats)

    st.write("**Distribution des variables numériques :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")
        st.pyplot(fig)

    st.write("**Boxplots :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
        st.pyplot(fig)

    st.write("**Matrice de corrélation :**")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def apply_fp_growth(df):
    st.subheader("Analyse par FP-Growth")

    required_cols = ['InvoiceNo', 'Description', 'Quantity']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Les colonnes {required_cols} sont nécessaires pour FP-Growth.")
        return

    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
    if freq_items.empty:
        st.warning("Aucun item fréquent trouvé avec le support minimal choisi.")
        return
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    if rules.empty:
        st.warning("Aucune règle d'association générée.")
        return
    st.write("Règles générées :")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

def apply_kmeans(df):
    st.subheader("Segmentation par K-means")

    features = df.select_dtypes(include='number').copy()
    if features.empty:
        st.error("❌ Pas de variables numériques disponibles pour K-means.")
        return

    k = st.slider("Choisissez le nombre de clusters", 2, 10, 4)
    X_scaled = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.write("Extrait des clusters avec variables sélectionnées :")
    st.dataframe(df.head())

    # Résumé des clusters
    st.write("Résumé des clusters :")
    cluster_summary = df.groupby('Cluster')[features.columns].mean().round(2)
    st.dataframe(cluster_summary)

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='Set2', ax=ax)
    ax.set_title("Projection 2D des clusters (PC1 vs PC2)")
    st.pyplot(fig)

def apply_rfm(df):
    st.subheader("Segmentation RFM")

    required_cols = ['Recence', 'Frequence', 'Montant']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Les colonnes {required_cols} sont nécessaires pour RFM.")
        return

    rfm = df[['Recence', 'Frequence', 'Montant']].copy()
    try:
        rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4,3,2,1])
        rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1,2,3,4])
        rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1,2,3,4])
    except ValueError as e:
        st.error(f"Erreur lors du découpage en quartiles: {e}")
        return

    rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)
    st.write("Extrait des scores RFM :")
    st.dataframe(rfm.head())

    fig, ax = plt.subplots()
    sns.histplot(rfm['RFM_Score'], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution du score RFM")
    st.pyplot(fig)

if uploaded_file:
    with st.spinner("Chargement des données..."):
        df, encoding = load_data(uploaded_file)
    st.success(f"Fichier chargé avec succès ✅ (encodage détecté : {encoding})")
    st.write("Aperçu des données :")
    st.dataframe(df.head())

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
