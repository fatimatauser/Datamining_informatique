import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import chardet

st.set_page_config(page_title="Analyse Client E-commerce", layout="wide")
st.title("📊 Analyse des données clients e-commerce")

# Sidebar
st.sidebar.header("🔽 Chargement des données")
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
    st.subheader("📈 Statistiques descriptives")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        st.warning("Aucune variable numérique détectée.")
        return

    st.write("📊 Distribution des variables numériques :")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution de {col}")
        st.pyplot(fig)

def apply_fp_growth(df):
    st.subheader("🔗 Analyse par FP-Growth")

    required_cols = {'InvoiceNo', 'Description', 'Quantity'}
    if not required_cols.issubset(df.columns):
        st.error(f"Les colonnes nécessaires {required_cols} sont absentes du fichier.")
        return

    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)

    if rules.empty:
        st.warning("Aucune règle d'association trouvée.")
    else:
        st.write("✅ Règles générées :")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

def apply_kmeans(df):
    st.subheader("📍 Segmentation par K-means")

    numeric_data = df.select_dtypes(include='number')
    if numeric_data.empty:
        st.error("Le jeu de données ne contient pas de colonnes numériques pour K-means.")
        return

    k = st.slider("Nombre de clusters", 2, 10, 3)
    X_scaled = StandardScaler().fit_transform(numeric_data)

    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

    st.success(f"{k} clusters générés.")
    st.dataframe(df[['Cluster'] + list(numeric_data.columns)].head())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='Set2', ax=ax)
    ax.set_title("Visualisation des clusters (2 premières dimensions)")
    st.pyplot(fig)

def apply_rfm(df):
    st.subheader("🎯 Segmentation RFM")

    required_cols = {'Recence', 'Frequence', 'Montant'}
    if not required_cols.issubset(df.columns):
        st.error(f"Les colonnes nécessaires {required_cols} sont absentes du fichier.")
        return

    rfm = df[['Recence', 'Frequence', 'Montant']].copy()
    rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1, 2, 3, 4])
    rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].astype(int).sum(axis=1)

    st.dataframe(rfm.head())

    fig, ax = plt.subplots()
    sns.histplot(rfm['RFM_Score'], bins=10, kde=True, color='green')
    ax.set_title("Distribution des scores RFM")
    st.pyplot(fig)

# Application principale
if uploaded_file:
    df, encoding = load_data(uploaded_file)
    st.success(f"Fichier chargé avec succès ✅ (encodage détecté : {encoding})")
    st.dataframe(df.head())

    model_choice = st.sidebar.radio("Choisir une analyse :", 
        ["📊 Statistiques descriptives", "🔗 FP-Growth", "📍 K-means", "🎯 RFM"])

    if model_choice == "📊 Statistiques descriptives":
        show_descriptive_stats(df)
    elif model_choice == "🔗 FP-Growth":
        apply_fp_growth(df)
    elif model_choice == "📍 K-means":
        apply_kmeans(df)
    elif model_choice == "🎯 RFM":
        apply_rfm(df)
else:
    st.info("Veuillez charger un fichier pour commencer.")
