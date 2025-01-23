import subprocess
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def harmonize_colors(image_path, harmony_type="complementary"):
    # Charger l'image
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extraire les canaux HSV
    h, s, v = cv2.split(image_hsv)

    if harmony_type == "complementary":
        h = (h + 90) % 180  # Rotation de 180 degrés (complémentaire)
    elif harmony_type == "analogous":
        h = (h + 30) % 180  # Rotation de 30 degrés
    elif harmony_type == "triadic":
        h = (h + 60) % 180
    elif harmony_type == "monochromatic":
        s = np.clip(s - 50, 0, 255)  # Réduction de la saturation

    # Combiner les canaux modifiés
    harmonized_hsv = cv2.merge([h, s, v])
    harmonized_image = cv2.cvtColor(harmonized_hsv, cv2.COLOR_HSV2BGR)

    return harmonized_image

# Tester avec une image
image_path = "test3.jpg"
harmonized = harmonize_colors(image_path, harmony_type="complementary")
cv2.imwrite("harmonized_image.jpg", harmonized)

def plot_histogram(image, title):
    colors = ("b", "g", "r")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f"Canal {color.upper()}")
    plt.title(title)
    plt.xlabel("Intensité des pixels")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.show()

# Affichage des histogrammes
original_image = cv2.imread(image_path)
plot_histogram(original_image, "Image originale")
plot_histogram(harmonized, "Image harmonisée")