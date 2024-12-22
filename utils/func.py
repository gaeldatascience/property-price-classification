print("Le module func.py a été chargé correctement.")
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
