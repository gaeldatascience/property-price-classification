def generate_bins(data, column, num_intervals=10, method='linear'):
    """
    Génère des intervalles (bins) pour une colonne donnée.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à utiliser pour générer les bins.
        num_intervals (int): Le nombre d'intervalles souhaités (par défaut 10).
        method (str): La méthode pour générer les bins :
                      - 'linear': Intervalles réguliers.
                      - 'quantile': Intervalles basés sur les quantiles.
    
    Returns:
        np.ndarray: Une liste des limites des bins.
    """
    import numpy as np
    if column not in data.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    if data[column].isnull().any():
        raise ValueError(f"La colonne '{column}' contient des valeurs nulles. Veuillez les traiter avant de générer les bins.")
    
    if method == 'linear':
        # Générer des intervalles réguliers
        bins = np.linspace(data[column].min(), data[column].max(), num_intervals + 1)
    elif method == 'quantile':
        # Générer des intervalles basés sur les quantiles
        bins = np.quantile(data[column], np.linspace(0, 1, num_intervals + 1))
    else:
        raise ValueError(f"Méthode inconnue : '{method}'. Utilisez 'linear' ou 'quantile'.")
    
    return bins

def process_property_data(df):
    import numpy as np
    """
    Traite un DataFrame contenant des informations sur les biens immobiliers en modifiant les colonnes
    'size' et 'land_size' en fonction du 'property_type'.
    
    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'property_type' et 'size'.
    
    Returns:
        pd.DataFrame: DataFrame modifié avec les ajustements sur 'size' et 'land_size'.
    """
    # Définir les limites de taille par type de propriété
    size_limits = {
        "maison": 400,
        "appartement": 250,
        "divers": 300,
        "villa": 400,
        "viager": 250,
        "chambre": 250,
        "duplex": 250,
        "manoir": 400,
        "loft": 400,
        "chalet": 250
    }

    # Appliquer les limites pour les types définis
    for property_type, max_size in size_limits.items():
        df.loc[(df["property_type"] == property_type) & (df["size"] > max_size), "size"] = np.nan

    # Traiter les cas spécifiques avec land_size
    land_size_types = ["terrain", "terrain à bâtir", "ferme", "propriété", "parking"]
    for property_type in land_size_types:
        df.loc[df["property_type"] == property_type, "land_size"] = df["size"]
        df.loc[df["property_type"] == property_type, "size"] = np.nan if property_type != "parking" else 0

    return df



from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Évalue un modèle de machine learning et affiche des métriques de performance, 
    un rapport de classification, la matrice de confusion, et les 10 principales caractéristiques importantes.

    Paramètres :
    - model : modèle entraîné (par exemple, RandomForestClassifier).
    - X_train : ensemble d'entraînement (features).
    - y_train : cibles de l'ensemble d'entraînement.
    - X_test : ensemble de test (features).
    - y_test : cibles de l'ensemble de test.
    """
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilité pour la classe positive
    y_pred = model.predict(X_test)
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc}")
    
    # Rapport de classification
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Matrice de Confusion")
    plt.show()
    
    # Importance des caractéristiques (si applicable)
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Affichage des 10 principales caractéristiques
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'][:10][::-1], feature_importances['Importance'][:10][::-1])
        plt.xlabel("Importance")
        plt.title("Top 10 des caractéristiques importantes")
        plt.show()
    else:
        print("Le modèle fourni n'a pas d'attribut 'feature_importances_'.")

def remplacer_code_postal(cp):
    """
    Remplace les codes postaux des arrondissements de Lyon, Marseille et Paris
    par les codes postaux principaux de leurs villes.

    Args:
        cp (str): Le code postal.

    Returns:
        str: Le code postal principal ou le code postal original.
    """
    if cp in [
        "69001", "69002", "69003", "69004", "69005", "69006", "69007", "69008", "69009"
    ]:
        return "69000"
    elif cp in [
        "13001", "13002", "13003", "13004", "13005", "13006", "13007", "13008", "13009",
        "13010", "13011", "13012", "13013", "13014", "13015", "13016"
    ]:
        return "13000"
    elif cp in [
        "75001", "75002", "75003", "75004", "75005", "75006", "75007", "75008", "75009",
        "75010", "75011", "75012", "75013", "75014", "75015", "75016", "75116", "75017",
        "75018", "75019", "75020"
    ]:
        return "75000"
    return cp