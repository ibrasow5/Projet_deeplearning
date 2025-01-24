import cv2
from matplotlib import pyplot as plt
import numpy as np

def harmonize_colors_complementary(image_path, output_path):
    # Charger l'image
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Décalage des teintes de 180° pour l'harmonie complémentaire
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + 90) % 180  # OpenCV HSV hue range is [0, 179]

    # Ajustement de la saturation (canal S) pour éviter des couleurs trop saturées
    img_hsv[:, :, 1] = cv2.normalize(img_hsv[:, :, 1], None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)

    # Ajustement de la luminosité (canal V) pour des contrastes équilibrés
    img_hsv[:, :, 2] = cv2.normalize(img_hsv[:, :, 2], None, alpha=70, beta=255, norm_type=cv2.NORM_MINMAX)

    # Reconversion en RGB
    img_harmonized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Mélange de l'image originale et harmonisée (blending)
    blended_image = cv2.addWeighted(img, 0.6, img_harmonized, 0.4, 0)

    # Sauvegarder l'image harmonisée mélangée
    cv2.imwrite(output_path, blended_image)

    return blended_image

# Exemple d'utilisation
image_path = "im1.jpg"
output_path = "image_harmonisee.jpg"
image_harmonisee = harmonize_colors_complementary(image_path, output_path)

# Afficher l'image originale et harmonisée avec OpenCV
original = cv2.imread(image_path)
harmonized = cv2.imread(output_path)
cv2.imshow("Image originale", original)
cv2.imshow("Image harmonisee", harmonized)
cv2.waitKey(0)
cv2.destroyAllWindows()


