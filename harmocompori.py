import cv2

def harmonize_colors_complementary(image_path, output_path):
    # Charger l'image
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Décalage des teintes de 180° pour l'harmonie complémentaire
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + 90) % 180  # OpenCV HSV hue range is [0, 179]

    # Reconversion en RGB
    img_harmonized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Sauvegarder l'image harmonisée
    cv2.imwrite(output_path, img_harmonized)

    return img_harmonized

# Exemple d'utilisation
image_path = "im1.jpg"
output_path = "image_harmonisee1.jpg"
image_harmonisee = harmonize_colors_complementary(image_path, output_path)

# Afficher l'image originale et harmonisée avec OpenCV
original = cv2.imread(image_path)
harmonized = cv2.imread(output_path)
cv2.imshow("Image originale", original)
cv2.imshow("Image harmonisee", harmonized)
cv2.waitKey(0)
cv2.destroyAllWindows()
