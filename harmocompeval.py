import cv2
from matplotlib import pyplot as plt

def plot_histograms(original_path, harmonized_path):
    # Charger les images en HSV
    original = cv2.imread(original_path)
    harmonized = cv2.imread(harmonized_path)
    
    if original is None or harmonized is None:
        print("Erreur lors du chargement des images.")
        return
    
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    harmonized_hsv = cv2.cvtColor(harmonized, cv2.COLOR_BGR2HSV)

    # Tracer les histogrammes pour H, S et V
    couleurs = ['Teinte', 'Saturation', 'Luminosité']
    for i, couleur in enumerate(couleurs):
        plt.figure(figsize=(10, 4))
        plt.hist(original_hsv[:, :, i].ravel(), bins=50, alpha=0.6, color='blue', label=f"Original {couleur}")
        plt.hist(harmonized_hsv[:, :, i].ravel(), bins=50, alpha=0.6, color='red', label=f"Harmonisé {couleur}")
        plt.title(f"Comparaison des histogrammes - {couleur}")
        plt.xlabel(f"Intensité de {couleur}")
        plt.ylabel("Fréquence")
        plt.legend()
        plt.show()  # Afficher les graphiques

# Définir les chemins des images
image_path = 'test3.jpg'
output_path = 'image_harmonisee.jpg'

# Évaluer les histogrammes pour l'image originale et harmonisée
plot_histograms(image_path, output_path)
