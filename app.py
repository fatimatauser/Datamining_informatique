
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import mlxtend
import squarify      
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Analyse e-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Essayer différents encodages
            try:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            except:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Veuillez uploader un fichier CSV ou Excel.")
            return None
       
        st.success("Données chargées avec succès!")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

def clean_data(df):
    """Nettoie les données"""
    # Suppression des lignes entièrement vides
    df = df.dropna(how='all')
   
    # Suppression des doublons
    df = df.drop_duplicates()
   
    # Suppression des transactions sans CustomerID
    if 'CustomerID' in df.columns:
        df = df[~df['CustomerID'].isna()]
   
    # Suppression des valeurs manquantes restantes
    df = df.dropna()
   
    # SOLUTION DEFINITIVE POUR LES DATES
    if 'InvoiceDate' in df.columns:
        # Convertir en string puis en datetime
        df['InvoiceDate'] = df['InvoiceDate'].astype(str)
       
        # Correction 1: Essayer plusieurs formats de date
        try:
            df['InvoiceDate'] = pd.to_datetime(
                df['InvoiceDate'],
                errors='coerce',
                infer_datetime_format=True
            )
        except:
            # Si échec, essayer manuellement les formats communs
            for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M']:
                try:
                    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format=fmt, errors='coerce')
                    break
                except:
                    continue
       
        # Supprimer les dates invalides
        df = df.dropna(subset=['InvoiceDate'])
   
    # Correction 2: Convertir les colonnes numériques problématiques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Convertir en float pour éviter les problèmes de type
        df[col] = pd.to_numeric(df[col], errors='coerce')
   
    # Ajout du montant total si possible
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Montant'] = df['Quantity'] * df['UnitPrice']
   
    # Correction 3: Supprimer les valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
   
    # Réinitialiser les index
    df.reset_index(drop=True, inplace=True)
   
    return df

def global_stats(df):
    """Calcule les statistiques globales du DataFrame"""
    stats = {
        "Indicateur": [
            "Nombre de variables",
            "Nombre d'observations",
            "Nombre de valeurs manquantes",
            "Pourcentage de valeurs manquantes",
            "Nombre de lignes dupliquées",
            "Pourcentage de lignes dupliquées",
            "Nombre de lignes entièrement vides",
            "Pourcentage de lignes vides",
            "Nombre de colonnes vides",
            "Pourcentage de colonnes vides"
        ],
        "Valeur": [
            df.shape[1],
            df.shape[0],
            df.isna().sum().sum(),
            "{:.2%}".format(df.isna().sum().sum() / df.size),
            df.duplicated().sum(),
            "{:.2%}".format(df.duplicated().sum() / len(df)),
            df.isna().all(axis=1).sum(),
            "{:.2%}".format(df.isna().all(axis=1).sum() / len(df)),
            df.isnull().all().sum(),
            "{:.2%}".format(df.isnull().all().sum() / df.shape[1])
        ]
    }

    return pd.DataFrame(stats)

def safe_display_dataframe(df, max_rows=10000):
    """Affiche un DataFrame de manière sécurisée"""
    if len(df) > max_rows:
        st.warning(f"Le DataFrame contient trop de lignes ({len(df)}). Affichage limité à {max_rows} lignes.")
        df = df.head(max_rows)
   
    # Conversion des types complexes en string
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Essayer de convertir les objets complexes en string
                df[col] = df[col].astype(str)
            except:
                # Si échec, supprimer la colonne
                st.warning(f"Colonne '{col}' supprimée car type non supporté: {df[col].dtype}")
                df = df.drop(col, axis=1)
   
    st.dataframe(df)

def show_descriptive_stats(df):
    """Affiche des statistiques descriptives complètes avec visualisations"""
    st.header("📊 Analyse Descriptive Complète")
   
    tab_stats, tab_dist, tab_box, tab_bivar = st.tabs([
        "Statistiques de Base",
        "Distributions",
        "Boîtes à Moustaches",
        "Analyse Bivariée"
    ])
   
    with tab_stats:
        st.subheader("Statistiques Summaries")
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.write("**Aperçu des Données**")
            safe_display_dataframe(df.head())
           
            st.write("**Types des Variables**")
            types_df = pd.DataFrame(df.dtypes, columns=['Type']).reset_index()
            types_df.columns = ['Variable', 'Type']
            safe_display_dataframe(types_df)
           
        with col2:
            st.write("**Statistiques Numériques**")
            # Filtrer uniquement les colonnes numériques
            num_cols = df.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                safe_display_dataframe(df[num_cols].describe())
            else:
                st.warning("Aucune colonne numérique trouvée")
           
            st.write("**Valeurs Manquantes**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ['Variable', 'Nombre']
            missing['Pourcentage'] = (missing['Nombre'] / len(df)) * 100
            safe_display_dataframe(missing)
   
    with tab_dist:
        st.subheader("Distributions des Variables")
       
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
       
        col1, col2 = st.columns(2)
       
        with col1:
            if num_cols:
                selected_num = st.selectbox("Variable numérique", num_cols, key="num_var")
               
                if selected_num:
                    # Vérifier qu'il y a des valeurs à afficher
                    if df[selected_num].notna().sum() > 0:
                        fig = px.histogram(
                            df,
                            x=selected_num,
                            nbins=50,
                            title=f"Distribution de {selected_num}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Aucune donnée valide pour la colonne {selected_num}")
            else:
                st.warning("Aucune variable numérique trouvée")
       
        with col2:
            if cat_cols:
                selected_cat = st.selectbox("Variable catégorielle", cat_cols, key="cat_var")
               
                if selected_cat:
                    # Limiter à un nombre raisonnable de catégories
                    unique_count = df[selected_cat].nunique()
                    if unique_count > 50:
                        st.warning(f"Trop de catégories ({unique_count}). Affichage limité aux 20 premières.")
                        top_cats = df[selected_cat].value_counts().head(20).index
                        count_data = df[df[selected_cat].isin(top_cats)][selected_cat].value_counts().reset_index()
                    else:
                        count_data = df[selected_cat].value_counts().reset_index()
                   
                    count_data.columns = ['Catégorie', 'Count']
                   
                    if not count_data.empty:
                        fig = px.bar(
                            count_data,
                            x='Catégorie',
                            y='Count',
                            title=f"Distribution de {selected_cat}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Aucune donnée à afficher")
            else:
                st.warning("Aucune variable catégorielle trouvée")
   
    with tab_box:
        st.subheader("Boîtes à Moustaches et Variance")
       
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_box = st.multiselect(
                "Variables numériques à comparer",
                num_cols,
                default=num_cols[:min(3, len(num_cols))],
                key="box_vars"
            )
           
            if selected_box:
                # Filtrer les colonnes avec des données valides
                valid_cols = [col for col in selected_box if df[col].notna().sum() > 0]
               
                if valid_cols:
                    # Créer un DataFrame avec uniquement les colonnes valides
                    plot_df = df[valid_cols].melt(value_vars=valid_cols).dropna()
                   
                    if not plot_df.empty:
                        fig = px.box(
                            plot_df,
                            x='variable',
                            y='value',
                            title="Comparaison des Distributions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                       
                        st.subheader("Matrice de Corrélation")
                        corr_matrix = df[valid_cols].corr().round(2)
                       
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Corrélations entre Variables"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Aucune donnée valide pour l'affichage")
                else:
                    st.warning("Aucune colonne valide sélectionnée")
        else:
            st.warning("Aucune variable numérique trouvée pour les boîtes à moustaches")
   
    with tab_bivar:
        st.subheader("Analyse Bivariée")
       
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
           
            with col1:
                x_var = st.selectbox("Variable X", num_cols, key="x_var")
            with col2:
                y_var = st.selectbox("Variable Y", num_cols, index=1, key="y_var")
           
            # Vérifier que les colonnes ont des données
            if df[[x_var, y_var]].notna().any().all():
                fig = px.scatter(
                    df,
                    x=x_var,
                    y=y_var,
                    title=f"Relation entre {x_var} et {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
               
                if num_cols:
                    corr = df[[x_var, y_var]].corr().iloc[0,1]
                    st.metric("Coefficient de Corrélation", f"{corr:.2f}")
            else:
                st.warning("Données insuffisantes pour générer le graphique")
        else:
            st.warning("Au moins 2 variables numériques requises pour l'analyse bivariée")

def perform_fpgrowth_analysis(df):
    """Effectue l'analyse FP-Growth"""
    # Vérification des colonnes nécessaires
    required_cols = ['InvoiceNo', 'Description', 'Quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
   
    if missing_cols:
        st.error(f"Colonnes manquantes pour FP-Growth: {', '.join(missing_cols)}")
        return
   
    with st.spinner("Préparation des données pour FP-Growth..."):
        try:
            # Création du panier
            basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].count().unstack().fillna(0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)
           
            # Supprimer les colonnes avec des noms manquants
            basket = basket.loc[:, ~basket.columns.isin([np.nan, None, ''])]
           
            # Limiter aux colonnes les plus fréquentes si trop nombreuses
            if len(basket.columns) > 100:
                st.warning("Trop de produits ({}). Utilisation des 100 plus fréquents.".format(len(basket.columns)))
                top_products = basket.sum().sort_values(ascending=False).head(100).index
                basket = basket[top_products]
        except Exception as e:
            st.error(f"Erreur lors de la préparation des données: {e}")
            return
   
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Support minimum", 0.01, 0.5, 0.01, 0.01, key="min_support")
    with col2:
        min_lift = st.slider("Lift minimum", 1.0, 10.0, 1.0, 0.1, key="min_lift")
   
    if st.button("Exécuter FP-Growth"):
        with st.spinner("Calcul des règles d'association..."):
            try:
                frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
               
                if frequent_itemsets.empty:
                    st.warning("Aucun itemset fréquent trouvé avec ce support minimum")
                    return
               
                rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
               
                if rules.empty:
                    st.warning("Aucune règle d'association trouvée avec ces paramètres")
                    return
               
                # Nettoyage des règles
                rules['pair_key'] = rules.apply(lambda row: tuple(sorted([row['antecedents'], row['consequents']])), axis=1)
                rules = rules.drop_duplicates(subset='pair_key')
                rules.drop(columns='pair_key', inplace=True)
               
                # Conversion en liste
                rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
               
                # Formatage pour l'affichage
                rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ", ".join(x))
                rules['consequents_str'] = rules['consequents'].apply(lambda x: ", ".join(x))
            except Exception as e:
                st.error(f"Erreur lors du calcul des règles: {e}")
                return
       
        st.success(f"FP-Growth terminé! {len(rules)} règles générées")
       
        # Affichage des résultats
        st.subheader("Top 20 des Règles d'Association")
        safe_display_dataframe(rules.sort_values(by='lift', ascending=False).head(20))
       
        # Visualisation
        st.subheader("Visualisation des Règles")
       
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Nombre de règles à visualiser", 5, 50, 10, key="top_n_fp")
           
            fig1 = px.scatter(
                rules.head(top_n),
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents_str', 'consequents_str'],
                title="Support vs Confiance"
            )
            st.plotly_chart(fig1, use_container_width=True)
       
        with col2:
            fig2 = px.bar(
                rules.sort_values(by='lift', ascending=False).head(top_n),
                x='lift',
                y='antecedents_str',
                color='consequents_str',
                orientation='h',
                title="Top Règles par Lift"
            )
            st.plotly_chart(fig2, use_container_width=True)
       
        # Téléchargement des résultats
        st.subheader("Télécharger les résultats")
        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les règles en CSV",
            data=csv,
            file_name="regles_association.csv",
            mime='text/csv'
        )

def perform_kmeans_analysis(df):
    """Effectue l'analyse K-means avec contrat de maintenance"""
    st.subheader("Sélection des Variables pour le Clustering")
   
    # Préparation des données RFM
    if 'InvoiceDate' in df.columns and 'InvoiceNo' in df.columns and 'Montant' in df.columns:
        # Convertir InvoiceDate en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
            try:
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            except:
                st.error("Impossible de convertir les dates")
                return
           
        date_ref = df['InvoiceDate'].max() + timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (date_ref - x.max()).days,
            'InvoiceNo': 'nunique',
            'Montant': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']
        df_rfm = rfm
    else:
        st.warning("Colonnes manquantes pour le calcul RFM. Utilisation de toutes les colonnes numériques.")
        df_rfm = df.select_dtypes(include=np.number)
   
    # Sélection des variables
    num_cols = df_rfm.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.multiselect(
        "Sélectionnez les variables pour le clustering",
        num_cols,
        default=num_cols[:min(3, len(num_cols))],
        key="kmeans_vars"
    )
   
    if not selected_cols:
        st.warning("Veuillez sélectionner au moins une variable numérique.")
        return
   
    # Standardisation des données
    scaler = StandardScaler()
   
    # Supprimer les valeurs manquantes et infinies
    df_rfm_clean = df_rfm[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
   
    if len(df_rfm_clean) == 0:
        st.error("Aucune donnée valide après nettoyage")
        return
       
    scaled_data = scaler.fit_transform(df_rfm_clean)
   
    # Détermination du nombre de clusters
    st.subheader("Détermination du Nombre Optimal de Clusters")
    max_clusters = st.slider("Nombre maximum de clusters à tester", 3, 15, 10, key="max_clusters")
   
    if st.button("Trouver le nombre optimal de clusters"):
        with st.spinner("Calcul en cours..."):
            wcss = []  # Within-Cluster Sum of Square
            silhouette_scores = []
           
            for i in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                    kmeans.fit(scaled_data)
                    wcss.append(kmeans.inertia_)
                   
                    if i > 1:  # Silhouette nécessite au moins 2 clusters
                        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
                except Exception as e:
                    st.error(f"Erreur avec {i} clusters: {e}")
                    break
           
            # Trouver le meilleur nombre de clusters basé sur le score de silhouette
            if silhouette_scores:
                best_n = np.argmax(silhouette_scores) + 2
            else:
                best_n = 3
                st.warning("Impossible de calculer le score de silhouette, utilisation par défaut de 3 clusters")
       
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Méthode du Coude")
            fig1, ax1 = plt.subplots()
            ax1.plot(range(2, max_clusters + 1), wcss, marker='o')
            ax1.set_title('Méthode du Coude')
            ax1.set_xlabel('Nombre de clusters')
            ax1.set_ylabel('WCSS')
            st.pyplot(fig1)
       
        with col2:
            st.subheader("Score de Silhouette")
            fig2, ax2 = plt.subplots()
            ax2.plot(range(2, len(silhouette_scores)+2), silhouette_scores, marker='o', color='green')
            ax2.set_title('Score de Silhouette')
            ax2.set_xlabel('Nombre de clusters')
            ax2.set_ylabel('Score de Silhouette')
            st.pyplot(fig2)
       
        st.success(f"Le nombre optimal de clusters est : {best_n} (score de silhouette: {silhouette_scores[best_n-2]:.3f})")
        st.session_state.best_n = best_n
        st.session_state.scaled_data = scaled_data
        st.session_state.df_rfm = df_rfm
        st.session_state.selected_cols = selected_cols
        st.session_state.scaler = scaler
        st.session_state.df_rfm_clean = df_rfm_clean
   
    # Application de K-means
    if 'best_n' in st.session_state:
        st.subheader("Application de K-means")
        n_clusters = st.slider(
            "Nombre de clusters",
            2, max_clusters, st.session_state.best_n,
            key="n_clusters"
        )
       
        if st.button("Exécuter K-means"):
            with st.spinner("Clustering en cours..."):
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
                clusters = kmeans.fit_predict(st.session_state.scaled_data)
                silhouette_avg = silhouette_score(st.session_state.scaled_data, clusters)
               
                # Ajout des clusters aux données
                df_clustered = st.session_state.df_rfm_clean.copy()
                df_clustered['Cluster'] = clusters
           
            st.success(f"Clustering terminé! Score de silhouette: {silhouette_avg:.3f}")
           
            # Visualisation des clusters
            st.subheader("Visualisation des Clusters")
           
            if len(st.session_state.selected_cols) >= 2:
                fig = px.scatter(
                    df_clustered,
                    x=st.session_state.selected_cols[0],
                    y=st.session_state.selected_cols[1],
                    color='Cluster',
                    title="Visualisation des Clusters",
                    hover_data=df_clustered.columns
                )
                st.plotly_chart(fig, use_container_width=True)
           
            # Profil des clusters
            st.subheader("Profil des Clusters")
            cluster_profile = df_clustered.groupby('Cluster').agg({
                'Recence': ['mean', 'min', 'max'],
                'Frequence': ['mean', 'min', 'max'],
                'Montant': ['mean', 'min', 'max'],
                'CustomerID': 'count'
            }).reset_index()
           
            safe_display_dataframe(cluster_profile)
           
            # Interprétation
            st.subheader("Interprétation des Clusters")
            for cluster in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                recence_moy = cluster_data['Recence'].mean()
                frequence_moy = cluster_data['Frequence'].mean()
                montant_moy = cluster_data['Montant'].mean()
               
                st.markdown(f"**Cluster {cluster}** (Nombre: {len(cluster_data)})")
                st.markdown(f"- Récence moyenne: {recence_moy:.1f} jours")
                st.markdown(f"- Fréquence moyenne: {frequence_moy:.1f} commandes")
                st.markdown(f"- Montant moyen: {montant_moy:.1f} €")
                st.markdown("---")
           
            # ===========================================
            # PARTIE CONTRAT DE MAINTENANCE (STABILITÉ TEMPORELLE)
            # ===========================================
            st.subheader("Contrat de Maintenance: Étude de Stabilité Temporelle")
           
            # Vérification des données temporelles
            if 'InvoiceDate' not in df.columns:
                st.warning("La colonne 'InvoiceDate' est nécessaire pour l'étude de stabilité temporelle.")
                return
           
            st.markdown("""
            ### Objectif
            Évaluer la stabilité du modèle de clustering au fil du temps en comparant les résultats
            sur différentes périodes temporelles à l'aide de l'Adjusted Rand Index (ARI).
            """)
           
            # Fonctions pour l'étude de stabilité
            def echantillonnage(df, n, pas=30, duree=180):
                date_min = df['InvoiceDate'].min()
                date_min = date_min + timedelta(days=pas * (n-1))
                date_max = date_min + timedelta(days=duree)
               
                sample_df = df[(df['InvoiceDate'] >= date_min) &
                            (df['InvoiceDate'] < date_max)].copy()
               
                st.info(f"Échantillon {n}: {sample_df.shape[0]} lignes ({date_min.strftime('%Y-%m-%d')} au {date_max.strftime('%Y-%m-%d')})")
                return sample_df

            def prepare_data(df):
                if 'Montant' not in df.columns:
                    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
                        df['Montant'] = df['Quantity'] * df['UnitPrice']
                    else:
                        st.error("Impossible de calculer le montant")
                        return None
               
                derniere_date = df['InvoiceDate'].max() + timedelta(days=1)
               
                rfm = df.groupby('CustomerID').agg({
                    'InvoiceDate': lambda x: (derniere_date - x.max()).days,
                    'InvoiceNo': 'nunique',
                    'Montant': 'sum'
                }).rename(columns={
                    'InvoiceDate': 'Recence',
                    'InvoiceNo': 'Frequence',
                    'Montant': 'MontantTotal'
                })
                return rfm

            if st.button("Lancer l'étude de stabilité temporelle"):
                with st.spinner("Création des échantillons temporels..."):
                    # Création des échantillons
                    samples = {}
                    for n in range(1, 10):
                        sample_df = echantillonnage(df, n)
                        sample_rfm = prepare_data(sample_df)
                        if sample_rfm is not None:
                            samples[f"B{n-1}"] = sample_rfm
                   
                    if not samples:
                        st.error("Aucun échantillon valide créé")
                        return
                   
                    st.session_state.samples = samples
                    st.success(f"{len(samples)} échantillons temporels créés avec succès!")
               
                # Préparation du modèle de référence
                with st.spinner("Création du modèle de référence..."):
                    scaler_ref = StandardScaler()
                    B0_scaled = scaler_ref.fit_transform(samples["B0"])
                    model0 = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                    model0.fit(B0_scaled)
                    st.session_state.model0 = model0
                    st.session_state.scaler_ref = scaler_ref
               
                # Fonction d'évaluation
                def entrainer_modele(data, modele_reference, scaler_ref):
                    data_scaled = scaler_ref.transform(data)
                    nouveau_modele = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                    nouveau_modele.fit(data_scaled)
                    labels_ref = modele_reference.predict(scaler_ref.transform(data))
                    labels_nouv = nouveau_modele.labels_
                    ari = adjusted_rand_score(labels_ref, labels_nouv)
                    return ari
               
                # Calcul des scores ARI
                with st.spinner("Calcul des scores de stabilité..."):
                    ARI_scores = []
                    sample_names = sorted(st.session_state.samples.keys())
                   
                    for i, name in enumerate(sample_names[1:], start=1):  # Commencer à B1
                        ari = entrainer_modele(
                            st.session_state.samples[name],
                            st.session_state.model0,
                            st.session_state.scaler_ref
                        )
                        ARI_scores.append(ari)
                        st.write(f"Échantillon B{i}: ARI = {ari:.4f}")
               
                st.session_state.ARI_scores = ARI_scores
           
            # Affichage des résultats de stabilité
            if 'ARI_scores' in st.session_state:
                st.subheader("Résultats de l'étude de stabilité")
               
                # Tableau des scores
                results_df = pd.DataFrame({
                    'Période': [f"B{i}" for i in range(1, len(st.session_state.ARI_scores)+1)],
                    'Score ARI': st.session_state.ARI_scores
                })
                safe_display_dataframe(results_df)
               
                # Graphique d'évolution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(st.session_state.ARI_scores)+1), st.session_state.ARI_scores, 'o-', markersize=8)
                ax.axhline(y=0.5, color='r', linestyle='--', label='Seuil de stabilité')
                ax.set_xlabel('Période')
                ax.set_ylabel('Score ARI')
                ax.set_title('Évolution de la Stabilité du Modèle')
                ax.set_xticks(range(1, len(st.session_state.ARI_scores)+1))
                ax.set_xticklabels([f"Période {i}" for i in range(1, len(st.session_state.ARI_scores)+1)])
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
               
                # Interprétation et recommandations
                st.subheader("Interprétation et Plan de Maintenance")
               
                # Analyse de la stabilité
                if st.session_state.ARI_scores:
                    min_ari = min(st.session_state.ARI_scores)
                    if min_ari > 0.7:
                        stability = "excellente"
                        recommendation = "Surveillance trimestrielle suffisante"
                        color = "green"
                    elif min_ari > 0.5:
                        stability = "bonne"
                        recommendation = "Surveillance mensuelle recommandée"
                        color = "orange"
                    elif min_ari > 0.3:
                        stability = "modérée"
                        recommendation = "Réévaluation bimestrielle nécessaire"
                        color = "orange"
                    else:
                        stability = "faible"
                        recommendation = "Réentraînement immédiat du modèle requis"
                        color = "red"
                   
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding: 0.5em 1em; background-color: #f8f9fa;">
                        <h4>Évaluation de la stabilité: <span style="color:{color};">{stability}</span></h4>
                        <p>Score ARI minimum: {min_ari:.4f}</p>
                        <p><strong>Recommandation:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Aucun score ARI disponible")
               
                # Plan de maintenance
                st.markdown("""
                ### Plan de Maintenance Proposé
               
                | Activité                         | Fréquence       | Responsable       |
                |----------------------------------|-----------------|-------------------|
                | Surveillance du score ARI        | Hebdomadaire    | Data Scientist    |
                | Analyse de la stabilité          | Mensuelle       | Data Scientist    |
                | Réévaluation du modèle           | Trimestrielle   | Équipe Analytics  |
                | Réentraînement du modèle         | Selon besoin    | Équipe Analytics  |
                | Rapport de performance           | Trimestriel     | Responsable Projet|
               
                **Seuils d'alerte:**
                - Avertissement: Score ARI < 0.5
                - Action immédiate: Score ARI < 0.3
               
                **Actions correctives:**
                1. Réentraîner le modèle avec les données récentes
                2. Réévaluer le nombre optimal de clusters
                3. Investiguer les changements dans le comportement des clients
                4. Mettre à jour les stratégies marketing en fonction des nouveaux segments
                """)

def perform_rfm_analysis(df):
    """Effectue l'analyse RFM"""
    # Vérification des colonnes nécessaires
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Montant']
    if 'Montant' not in df.columns:
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['Montant'] = df['Quantity'] * df['UnitPrice']
        else:
            st.error("Impossible de calculer le montant: colonnes manquantes")
            return
   
    missing_cols = [col for col in required_cols if col not in df.columns]
   
    if missing_cols:
        st.error(f"Colonnes manquantes pour l'analyse RFM: {', '.join(missing_cols)}")
        return
   
    # Convertir InvoiceDate en datetime si nécessaire
    if 'InvoiceDate' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        except:
            st.error("Impossible de convertir les dates")
            return
   
    # Calcul RFM
    date_ref = df['InvoiceDate'].max() + timedelta(days=1)
   
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (date_ref - x.max()).days,
        'InvoiceNo': 'nunique',
        'Montant': 'sum'
    }).reset_index()
   
    rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']
   
    # Calcul des quartiles
    quantiles = rfm[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()
   
    # Fonctions de scoring
    def r_score(x):
        if x <= quantiles['Recence'][0.25]:
            return 4
        elif x <= quantiles['Recence'][0.5]:
            return 3
        elif x <= quantiles['Recence'][0.75]:
            return 2
        else:
            return 1
   
    def fm_score(x, var):
        if x <= quantiles[var][0.25]:
            return 1
        elif x <= quantiles[var][0.5]:
            return 2
        elif x <= quantiles[var][0.75]:
            return 3
        else:
            return 4
   
    # Application des scores
    rfm['R'] = rfm['Recence'].apply(r_score)
    rfm['F'] = rfm['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
    rfm['M'] = rfm['Montant'].apply(lambda x: fm_score(x, 'Montant'))
    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
   
    # Segmentation
    contrat = {
        r'[1-2][1-2]': 'en hibernation',
        r'[1-2][3-4]': 'à risque',
        r'[1-2]5': 'à ne pas perdre',
        r'3[1-2]': 'sur le point de dormir',
        r'33': 'nécessite de l\'attention',
        r'[3-4][4-5]': 'clients fidèles',
        r'41': 'prometteurs',
        r'51': 'nouveaux clients',
        r'[4-5][2-3]': 'clients potentiellement fidèles',
        r'5[4-5]': 'champions'
    }
   
    rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(contrat, regex=True)
   
    # Affichage des résultats
    st.subheader("Résultats de l'analyse RFM")
    safe_display_dataframe(rfm.head(10))
   
    # Visualisation
    st.subheader("Visualisation des Segments RFM")
   
    # Distribution des segments
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Nombre de Clients']
   
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            segment_counts,
            x='Segment',
            y='Nombre de Clients',
            title="Répartition des Segments RFM"
        )
        st.plotly_chart(fig1, use_container_width=True)
   
    with col2:
        fig2 = px.pie(
            segment_counts,
            names='Segment',
            values='Nombre de Clients',
            title="Distribution des Segments RFM"
        )
        st.plotly_chart(fig2, use_container_width=True)
   
    # Treemap
    st.subheader("Treemap des Segments RFM")
    plt.figure(figsize=(12, 8))
    colors = ['#f94144', '#f3722c', '#f9c74f', '#90be6d', '#43aa8b',
              '#577590', '#277da1', '#4d908e', '#f9844a', '#8ac926']
   
    squarify.plot(
        sizes=segment_counts['Nombre de Clients'],
        label=segment_counts['Segment'],
        color=colors,
        alpha=.9,
        text_kwargs={'fontsize': 12, 'weight': 'bold'}
    )
   
    plt.title("Segmentation RFM des Clients", fontsize=16)
    plt.axis('off')
    st.pyplot(plt)
   
    # Recommandations
    st.subheader("Recommandations par Segment")
    recommendations = {
        'champions': "Récompenser, offres exclusives, programmes VIP",
        'clients fidèles': "Fidélisation, offres personnalisées",
        'clients potentiellement fidèles': "Encourager à acheter plus",
        'à ne pas perdre': "Offres spéciales, attention particulière",
        'prometteurs': "Encourager à devenir fidèles",
        'à risque': "Campagnes de réactivation",
        'sur le point de dormir': "Relances par email, offres de retour",
        'nécessite de l\'attention': "Offres ciblées, rappels",
        'nouveaux clients': "Bienvenue, guide d'utilisation",
        'en hibernation': "Campagnes agressives de réactivation"
    }
   
    rec_df = pd.DataFrame.from_dict(recommendations, orient='index', columns=['Recommandation'])
    safe_display_dataframe(rec_df)

# Interface principale
def main():
    st.title("🛒 Plateforme d'Analyse e-commerce")
    st.markdown("""
    Cette application permet d'analyser les données clients d'un site e-commerce à l'aide de trois approches:
    - **Règles d'association (FP-Growth)**: Découvrir quels produits sont fréquemment achetés ensemble
    - **Segmentation (K-means)**: Grouper les clients en clusters similaires
    - **Analyse RFM**: Segmenter les clients basé sur la Récence, Fréquence et Montant des achats
    """)
   
    # Chargement des données
    st.sidebar.header("Chargement des Données")
    uploaded_file = st.sidebar.file_uploader("Uploader votre fichier de données (CSV ou Excel)", type=['csv', 'xlsx'])
   
    df = None
    df_clean = None
   
    if uploaded_file is not None:
        df = load_data(uploaded_file)
       
        if df is not None:
            # Onglets principaux
            tab_stats, tab_fp, tab_kmeans, tab_rfm = st.tabs([
                "Statistiques",
                "FP-Growth",
                "K-means",
                "RFM"
            ])
           
            with tab_stats:
                # Statistiques globales
                st.header("Statistiques Globales")
                stats_df = global_stats(df)
                safe_display_dataframe(stats_df)
               
                # Nettoyage des données
                if st.button("Nettoyer les données", key="clean_data_btn"):
                    with st.spinner("Nettoyage en cours..."):
                        df_clean = clean_data(df)
                    st.success("Données nettoyées avec succès!")
                    st.session_state.df_clean = df_clean
               
                # Afficher les stats descriptives si données nettoyées
                if 'df_clean' in st.session_state:
                    st.header("Données Après Nettoyage")
                    st.write(f"Dimensions: {st.session_state.df_clean.shape[0]} lignes, {st.session_state.df_clean.shape[1]} colonnes")
                    show_descriptive_stats(st.session_state.df_clean)
                else:
                    st.info("Cliquez sur 'Nettoyer les données' pour commencer l'analyse")
           
            # Onglet FP-Growth
            with tab_fp:
                st.header("Analyse par Règles d'Association (FP-Growth)")
                if 'df_clean' in st.session_state:
                    perform_fpgrowth_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
           
            # Onglet K-means
            with tab_kmeans:
                st.header("Segmentation des Clients (K-means)")
                if 'df_clean' in st.session_state:
                    perform_kmeans_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
           
            # Onglet RFM
            with tab_rfm:
                st.header("Analyse RFM (Récence, Fréquence, Montant)")
                if 'df_clean' in st.session_state:
                    perform_rfm_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
    else:
        st.info("Veuillez uploader un fichier de données pour commencer l'analyse")

if __name__ == "__main__":
    main()
