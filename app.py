import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import chardet

# --- Config de la page ---
st.set_page_config(page_title="Analyse Client e-commerce", layout="wide", initial_sidebar_state="expanded")

# --- Titre principal ---
st.title("📊 Application d'analyse e-commerce")
st.markdown("""
Bienvenue dans cette application d'analyse de données e-commerce.  
Chargez vos données clients et choisissez le modèle d'analyse adapté à vos besoins : statistiques descriptives, règles d'association (FP-Growth), segmentation K-means ou segmentation RFM.
""")

# --- Sidebar ---
st.sidebar.header("🔎 Chargement et sélection")

uploaded_file = st.sidebar.file_uploader(
    "Uploader un fichier CSV ou Excel",
    type=["csv", "xlsx"],
    help="Formats acceptés : CSV, Excel (.xlsx)."
)

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
    st.subheader("📈 Statistiques descriptives")
    st.write(df.describe())
    st.markdown("**Distribution des variables numériques :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color='cornflowerblue')
        ax.set_title(f'Distribution de {col}')
        st.pyplot(fig)

def apply_fp_growth(df):
    st.subheader("🔗 Analyse par FP-Growth")
    required_cols = ['InvoiceNo', 'Description', 'Quantity']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Les colonnes {required_cols} sont requises pour FP-Growth.")
        return
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
    if freq_items.empty:
        st.warning("Aucun item fréquent trouvé avec ce support minimum.")
        return
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    if rules.empty:
        st.warning("Aucune règle d'association générée.")
        return
    st.write("Règles générées :")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))

def apply_kmeans(df):
    st.subheader("📊 Segmentation par K-means")
    features = df.select_dtypes(include='number')
    if features.shape[1] < 2:
        st.error("Le jeu de données doit contenir au moins 2 variables numériques pour appliquer K-means.")
        return
    k = st.slider("Choisissez le nombre de clusters", 2, 10, 4)
    X_scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    df['Cluster'] = clusters
    st.write("Extrait des clusters :")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='Set2', ax=ax)
    ax.set_title("Visualisation des clusters (sur les 2 premières variables numériques)")
    st.pyplot(fig)

def apply_rfm(df):
    st.subheader("🛍️ Segmentation RFM")
    required_cols = ['Recence', 'Frequence', 'Montant']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Les colonnes {required_cols} sont requises pour l'analyse RFM.")
        return
    rfm = df[required_cols].copy()
    rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1,2,3,4])
    rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1,2,3,4])
    rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)
    st.write("Extrait des scores RFM :")
    st.dataframe(rfm.head())

    fig, ax = plt.subplots()
    sns.histplot(rfm['RFM_Score'], bins=10, kde=True, color='seagreen', ax=ax)
    ax.set_title("Distribution des scores RFM")
    st.pyplot(fig)

if uploaded_file:
    try:
        df, encoding = load_data(uploaded_file)
        st.success(f"Fichier chargé avec succès ✅ (Encodage détecté : {encoding})")
        st.markdown("### Aperçu des données :")
        st.dataframe(df.head())
        
        model_choice = st.sidebar.selectbox(
            "Choisir une analyse",
            ["Statistiques descriptives", "FP-Growth", "K-means", "RFM"]
        )
        if model_choice == "Statistiques descriptives":
            show_descriptive_stats(df)
        elif model_choice == "FP-Growth":
            apply_fp_growth(df)
        elif model_choice == "K-means":
            apply_kmeans(df)
        elif model_choice == "RFM":
            apply_rfm(df)
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
else:
    st.info("📁 Veuillez charger un fichier pour commencer.")

# --- Section À propos ---
with st.sidebar.expander("ℹ️ À propos"):
    st.markdown("""
    **Application développée avec Streamlit**  
    - Auteur : Toi 😎  
    - Projet Data Mining e-commerce  
    - Chargement, analyse des règles d'association, clustering et segmentation RFM.  
    """)

