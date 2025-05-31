import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import chardet
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import base64

# Configuration de la page
st.set_page_config(
    page_title="Analyse Client e-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #F9F9F9;
        background-image: linear-gradient(to bottom, #ffffff, #e6f0ff);
    }
    h1 {
        color: #1E3A8A;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #EFF6FF;
        background-image: linear-gradient(to bottom, #ffffff, #d1e0ff);
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stSelectbox>div>div>select {
        background-color: #EFF6FF;
        border: 1px solid #1E3A8A;
    }
    .css-1aumxhk {
        background-color: #DBEAFE;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-image {
        display: block;
        margin: 0 auto;
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        padding: 15px;
        font-size: 0.9em;
        color: #555555;
        margin-top: 30px;
        border-top: 1px solid #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour encoder une image en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Image encodée en base64 (remplacer par votre chemin d'image si nécessaire)
# Si vous voulez utiliser une image locale, remplacez le code ci-dessous par:
# header_image = get_base64_image("chemin/vers/votre/image.jpg")
# Puis utilisez: st.markdown(f'<img src="data:image/png;base64,{header_image}" class="header-image">', unsafe_allow_html=True)

# Utilisation d'une image en ligne pour l'en-tête
HEADER_IMAGE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/images/ecommerce-analytics.png"

# Titre principal avec image d'en-tête
st.markdown(f'<img src="{HEADER_IMAGE_URL}" class="header-image" style="max-height:300px">', unsafe_allow_html=True)
st.title("📊 Plateforme d'Analyse Client e-commerce")
st.markdown("***Analyse avancée des comportements clients et segmentation marketing***")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration des données")
    st.image("https://cdn-icons-png.flaticon.com/512/3594/3594436.png", width=80)
    uploaded_file = st.file_uploader("Télécharger un fichier", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("🔍 Méthodes d'analyse")
    model_choice = st.selectbox("Sélectionnez une technique:", 
                               ["Statistiques descriptives", 
                                "Analyse de panier (FP-Growth)", 
                                "Segmentation client (K-means)", 
                                "Analyse RFM"])
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info("Cette application permet d'analyser les données clients e-commerce à l'aide de différentes méthodes statistiques et de ML.")

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
    st.header("📈 Statistiques descriptives")
    
    with st.expander("Aperçu des données"):
        st.write(f"**Dimensions du dataset :** {df.shape[0]} lignes × {df.shape[1]} colonnes")
        st.dataframe(df.head(3).style.set_properties(**{'background-color': '#EFF6FF'}))
    
    with st.expander("Résumé statistique"):
        st.write(df.describe().T.style.background_gradient(cmap='Blues'))
        
    with st.expander("Distribution des variables"):
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            col = st.selectbox("Choisir une variable numérique", num_cols)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df[col], kde=True, color='#1E3A8A')
            plt.title(f'Distribution de {col}')
            st.pyplot(fig)
        else:
            st.warning("Aucune variable numérique détectée")
            
    with st.expander("Corrélations"):
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
            plt.title('Matrice de corrélation')
            st.pyplot(fig)
        else:
            st.warning("Pas assez de variables numériques pour la matrice de corrélation")

def apply_fp_growth(df):
    st.header("🛒 Analyse de panier (FP-Growth)")
    
    with st.expander("Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            transaction_col = st.selectbox("Colonne transaction", df.columns)
        with c2:
            item_col = st.selectbox("Colonne produit", df.columns)
            
        min_support = st.slider("Support minimum", 0.01, 0.3, 0.02, 0.01)
        min_threshold = st.slider("Seuil de confiance minimum", 0.1, 1.0, 0.5, 0.05)
    
    try:
        # Création des transactions sous forme de listes d'articles
        transactions = df.groupby(transaction_col)[item_col].apply(list).tolist()
        
        # Encodage des transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        with st.spinner("Calcul des règles d'association..."):
            freq_items = fpgrowth(basket, min_support=min_support, use_colnames=True)
            
            if freq_items.empty:
                st.warning("Aucun ensemble fréquent trouvé. Essayez de réduire le support minimum.")
                return
                
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_threshold)
            
        st.success(f"{len(rules)} règles générées avec succès!")
        
        st.subheader("Top 10 des règles")
        top_rules = rules.sort_values('lift', ascending=False).head(10)
        
        # Formatage des règles pour l'affichage
        top_rules_display = top_rules.copy()
        top_rules_display['antecedents'] = top_rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules_display['consequents'] = top_rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.dataframe(top_rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].style.background_gradient(cmap='Blues'))
        
        st.subheader("Visualisation des règles")
        if not rules.empty:
            top_rules['rule'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x))) + " => " + top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            top_rules['rule'] = top_rules['rule'].str.wrap(50)  # Wrapping pour une meilleure lisibilité
            
            # Graphique 1: Nuage de points des règles
            fig = px.scatter(
                top_rules.head(20), 
                x='support', 
                y='confidence', 
                size='lift', 
                color='lift',
                hover_name='rule',
                labels={
                    'support': 'Support',
                    'confidence': 'Confiance'
                },
                color_continuous_scale='blues',
                height=600,
                title="Relations entre support, confiance et lift"
            )
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14))
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique 2: Top 10 des règles par lift
            top_rules_sorted = top_rules.sort_values('lift', ascending=False).head(10)
            fig2 = px.bar(
                top_rules_sorted,
                x='rule',
                y='lift',
                color='lift',
                color_continuous_scale='blues',
                labels={'rule': 'Règles', 'lift': 'Lift'},
                title="Top 10 des règles par valeur de Lift"
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Aucune règle à afficher")
        
    except Exception as e:
        st.error(f"Erreur dans l'analyse : {str(e)}")
        st.exception(e)

def apply_kmeans(df):
    st.header("👥 Segmentation client (K-means)")
    
    with st.expander("Paramètres", expanded=True):
        k = st.slider("Nombre de clusters", 2, 10, 4)
        num_cols = df.select_dtypes(include='number').columns
        features = st.multiselect("Variables à inclure", num_cols, default=num_cols[:min(2, len(num_cols)] if len(num_cols) > 0 else [])
    
    if len(features) < 2:
        st.warning("Sélectionnez au moins 2 variables")
        return
        
    # Création d'un dataframe temporaire sans valeurs manquantes
    temp_df = df[features].dropna().copy()
    
    if temp_df.empty:
        st.error("Aucune donnée disponible après suppression des valeurs manquantes.")
        return
        
    X = temp_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    with st.spinner("Création des clusters..."):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = model.fit_predict(X_scaled)
        temp_df['Cluster'] = clusters
        
        # Fusion avec le dataframe original
        clustered_df = df.copy()
        clustered_df = clustered_df.merge(temp_df[['Cluster']], left_index=True, right_index=True, how='left', suffixes=('', '_y'))
    
    st.success(f"{k} clusters créés avec succès! ({temp_df.shape[0]} clients segmentés)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Distribution des clusters")
        cluster_counts = temp_df['Cluster'].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.pie(cluster_counts, 
                labels=cluster_counts.index, 
                colors=sns.color_palette("Blues", k),
                autopct='%1.1f%%',
                startangle=90,
                shadow=True)
        ax1.axis('equal')
        st.pyplot(fig1)
        
        st.subheader("Caractéristiques des clusters")
        cluster_means = temp_df.groupby('Cluster')[features].mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
        
    with c2:
        st.subheader("Visualisation des clusters")
        
        if len(features) == 2:
            # Visualisation directe si 2 features
            fig2 = px.scatter(
                temp_df, 
                x=features[0], 
                y=features[1], 
                color='Cluster',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hover_data=features,
                title="Visualisation 2D des clusters"
            )
            fig2.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # PCA pour la réduction de dimension si plus de 2 features
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            temp_df['PCA1'] = pca_result[:, 0]
            temp_df['PCA2'] = pca_result[:, 1]
            
            fig2 = px.scatter(
                temp_df, 
                x='PCA1', 
                y='PCA2', 
                color='Cluster',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hover_data=features,
                title="Projection PCA (2D) des clusters"
            )
            fig2.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig2, use_container_width=True)
            
            # Variance expliquée
            var_ratio = pca.explained_variance_ratio_
            st.caption(f"Variance expliquée : PCA1 = {var_ratio[0]*100:.1f}%, PCA2 = {var_ratio[1]*100:.1f}%")
            
            # Graphique 3D optionnel
            if len(features) > 2:
                st.subheader("Visualisation 3D")
                pca3d = PCA(n_components=3)
                pca3d_result = pca3d.fit_transform(X_scaled)
                temp_df['PCA1_3d'] = pca3d_result[:, 0]
                temp_df['PCA2_3d'] = pca3d_result[:, 1]
                temp_df['PCA3_3d'] = pca3d_result[:, 2]
                
                fig3d = px.scatter_3d(
                    temp_df,
                    x='PCA1_3d',
                    y='PCA2_3d',
                    z='PCA3_3d',
                    color='Cluster',
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hover_data=features,
                    title="Projection 3D des clusters"
                )
                st.plotly_chart(fig3d, use_container_width=True)

def apply_rfm(df):
    st.header("💰 Analyse RFM")
    
    with st.expander("Configuration", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            recency_col = st.selectbox("Colonne récence", df.columns)
        with cols[1]:
            frequency_col = st.selectbox("Colonne fréquence", df.columns)
        with cols[2]:
            monetary_col = st.selectbox("Colonne montant", df.columns)
    
    try:
        # Création du dataframe RFM
        rfm = df[[recency_col, frequency_col, monetary_col]].copy()
        rfm.columns = ['Recence', 'Frequence', 'Montant']
        
        # Conversion en numérique et gestion des erreurs
        for col in ['Recence', 'Frequence', 'Montant']:
            rfm[col] = pd.to_numeric(rfm[col], errors='coerce')
        
        # Suppression des valeurs manquantes
        rfm = rfm.dropna()
        
        if rfm.empty:
            st.error("Aucune donnée valide après nettoyage.")
            return
            
        # Calcul des scores RFM
        rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4, 3, 2, 1])
        rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1, 2, 3, 4])
        rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1, 2, 3, 4])
        
        # Conversion en numérique
        rfm['R'] = rfm['R'].astype(int)
        rfm['F'] = rfm['F'].astype(int)
        rfm['M'] = rfm['M'].astype(int)
        
        rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)
        rfm['Segment'] = rfm['RFM_Score'].apply(assign_segment)
        
        st.subheader("Scores RFM")
        st.dataframe(rfm.head().style.background_gradient(cmap='Blues'))
        
        st.subheader("Distribution des segments RFM")
        segment_counts = rfm['Segment'].value_counts()
        fig1 = px.bar(
            segment_counts,
            x=segment_counts.index,
            y=segment_counts.values,
            color=segment_counts.values,
            color_continuous_scale='blues',
            labels={'x': 'Segment RFM', 'y': 'Nombre de clients'},
            text=segment_counts.values,
            title="Répartition des clients par segment RFM"
        )
        fig1.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Analyse des segments")
        segment_analysis = rfm.groupby('Segment').agg({
            'Recence': 'mean',
            'Frequence': 'mean',
            'Montant': 'mean',
            'RFM_Score': 'mean'
        }).reset_index()
        segment_analysis.columns = ['Segment', 'Recence moyenne', 'Frequence moyenne', 'Montant moyen', 'Score RFM moyen']
        
        fig2 = px.scatter(
            segment_analysis,
            x='Recence moyenne',
            y='Montant moyen',
            size='Frequence moyenne',
            color='Score RFM moyen',
            hover_name='Segment',
            size_max=60,
            color_continuous_scale='blues',
            title="Analyse des segments RFM"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Carte thermique RFM")
        rfm_agg = rfm.groupby('Segment').agg({
            'Recence': 'mean',
            'Frequence': 'mean',
            'Montant': 'mean'
        })
        fig3 = px.imshow(
            rfm_agg.T,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale='blues',
            labels=dict(x="Segment", y="Métrique", color="Valeur"),
            title="Moyennes RFM par segment"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Recommandations par segment")
        recommendations = {
            "Champions": "Offres premium, programme fidélité, nouveaux produits",
            "Fidèles": "Cross-selling, offres de réengagement, rabais",
            "Potentiels": "Promotions ciblées, paniers abandonnés, contenu éducatif",
            "Nouveaux": "Bienvenue, tutoriels, première offre spéciale",
            "Dormants": "Campagnes de réactivation, offres spéciales, enquêtes"
        }
        
        rec_df = pd.DataFrame({
            'Segment': recommendations.keys(),
            'Recommandation': recommendations.values()
        })
        st.table(rec_df.style.set_properties(**{'background-color': '#DBEAFE'}))
        
    except Exception as e:
        st.error(f"Erreur dans le calcul RFM : {str(e)}")
        st.exception(e)

def assign_segment(score):
    if score >= 10:
        return "Champions"
    elif score >= 8:
        return "Fidèles"
    elif score >= 6:
        return "Potentiels"
    elif score >= 4:
        return "Nouveaux"
    else:
        return "Dormants"

# Page principale
if uploaded_file:
    df, encoding = load_data(uploaded_file)
    
    st.success(f"✅ Données chargées ({df.shape[0]} lignes, {df.shape[1]} colonnes) | Encodage: {encoding}")
    st.markdown("---")
    
    if model_choice == "Statistiques descriptives":
        show_descriptive_stats(df)
    elif model_choice == "Analyse de panier (FP-Growth)":
        apply_fp_growth(df)
    elif model_choice == "Segmentation client (K-means)":
        apply_kmeans(df)
    elif model_choice == "Analyse RFM":
        apply_rfm(df)
else:
    st.info("ℹ️ Veuillez télécharger un fichier CSV ou Excel pour commencer l'analyse")
    st.image("https://raw.githubusercontent.com/plotly/datasets/master/images/ecommerce-dashboard.png", width=600)
    st.markdown("""
    <div style="background-color:#DBEAFE; padding:20px; border-radius:10px; margin-top:20px;">
        <h3 style="color:#1E3A8A;">Guide d'utilisation:</h3>
        <ol>
            <li>Téléchargez un fichier de données via le panneau latéral</li>
            <li>Sélectionnez une méthode d'analyse</li>
            <li>Configurez les paramètres spécifiques</li>
            <li>Explorez les résultats visuels</li>
        </ol>
        <p><i>Exemple de données compatibles : données transactionnelles e-commerce</i></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>📱 Application développée avec Streamlit | © 2023 - Analyse Client e-commerce</p>
    <p style="font-size:0.8em;">Plateforme d'analyse avancée pour optimiser votre stratégie e-commerce</p>
</div>
""", unsafe_allow_html=True)
