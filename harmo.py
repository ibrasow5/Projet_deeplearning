import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# Fonction pour afficher l'image dans l'interface graphique
def display_image(image_tensor, label):
    image = tf.squeeze(image_tensor, axis=0).numpy()  # Supprimer la dimension batch
    image = np.array(image * 255, dtype=np.uint8)  # Reconvertir entre 0 et 255
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)
    label.config(image=image_tk)
    label.image = image_tk

# Fonction pour charger l'image depuis le fichier
def load_image(image_path, target_size=(150, 150)):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normaliser l'image
        img_array = img_array.astype(np.float32)
        img_tensor = tf.expand_dims(img_array, axis=0)
        return img_tensor
    except Exception as e:
        print(f"Erreur : {e}")
        return None

# Fonction pour appliquer le transfert de style
def style_transfer(content_image, style_image, content_weight=1e4, style_weight=1e-2, epochs=10):
    # Charger le modèle VGG-19 pré-entraîné sans la couche de classification
    vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # Redimensionner les images de contenu et de style à la taille attendue par VGG-19 (224x224)
    content_image = tf.image.resize(content_image, (224, 224))  # Redimensionner à 224x224
    style_image = tf.image.resize(style_image, (224, 224))      # Redimensionner à 224x224

    # Prétraiter les images pour VGG-19 (normalisation)
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

    # Ajouter une dimension pour le batch (VGG-19 attend un batch de taille 1)
    content_image = tf.expand_dims(content_image, axis=0)  # Forme (1, 224, 224, 3)
    style_image = tf.expand_dims(style_image, axis=0)      # Forme (1, 224, 224, 3)

    # Extraire les caractéristiques de l'image de contenu et de style à partir de VGG-19
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    # Extraire les activations des couches choisies
    content_output = vgg_model.get_layer(content_layer).output
    style_outputs = [vgg_model.get_layer(layer).output for layer in style_layers]

    # Créer un modèle avec l'entrée d'image et les sorties des couches spécifiées
    model = tf.keras.Model(inputs=vgg_model.input, outputs=[content_output] + style_outputs)

    # Passer les images de contenu et de style dans le modèle pour obtenir les activations
    activations = model(content_image)

    # Extraire les activations de l'image de contenu et de style
    content_output = activations[0]
    style_outputs = activations[1:]

    # Squeeze the content_image to match the shape of content_output
    content_image = tf.squeeze(content_image, axis=0)

    # Calculer la perte de contenu
    content_loss = content_weight * tf.reduce_mean(tf.square(content_output - content_image))

    # Calculer la perte de style
    style_loss = 0
    for style_output in style_outputs:
        # Resize style_image to match the shape of style_output
        resized_style_image = tf.image.resize(style_image, style_output.shape[1:3])
        style_loss += tf.reduce_mean(tf.square(style_output - resized_style_image))

    # Combiner les pertes
    total_loss = content_loss + style_loss
    return total_loss

def apply_style_transfer():
    content_image = load_image(content_path.get())
    style_image = load_image(style_path.get())

    if content_image is None or style_image is None:
        messagebox.showerror("Erreur", "Erreur de chargement des images.")
        return

    # Paramètres
    content_weight = content_weight_slider.get()
    style_weight = style_weight_slider.get()
    epochs = int(epochs_entry.get())

    # Appliquer le transfert de style
    result_image = style_transfer(content_image, style_image, content_weight=content_weight, style_weight=style_weight, epochs=epochs)
    
    # Afficher l'image générée
    display_image(result_image, result_image_label)
    
    # Sauvegarder l'image générée avec la taille d'origine
    original_size = Image.open(content_path.get()).size
    save_path = os.path.join("generated_image.jpg")
    save_image(result_image, save_path, original_size)
    messagebox.showinfo("Succès", f"Image générée et sauvegardée à {save_path}")

# Fonction pour sauvegarder l'image avec la taille d'origine
def save_image(image_tensor, save_path, original_size):
    image = tf.squeeze(image_tensor, axis=0).numpy() * 255
    image = np.array(image, dtype=np.uint8)
    img = Image.fromarray(image)

    # Redimensionner l'image générée à la taille d'origine avant de la sauvegarder
    img = img.resize(original_size)
    img.save(save_path)

# Créer l'interface graphique principale
root = tk.Tk()
root.title("Transfert de Style")
root.geometry("800x800")
root.config(bg="#D3D3D3")  # Fond gris clair

# Frame pour l'affichage des images
image_frame = tk.Frame(root, bg="#D3D3D3")
image_frame.pack(pady=20)

# Labels pour afficher les images
content_image_label = tk.Label(image_frame, bg="#D3D3D3")
content_image_label.pack(side=tk.LEFT, padx=10)

style_image_label = tk.Label(image_frame, bg="#D3D3D3")
style_image_label.pack(side=tk.LEFT, padx=10)

result_image_label = tk.Label(root, bg="#D3D3D3")
result_image_label.pack(pady=20)

# Interface pour choisir les images de contenu et de style
content_label = tk.Label(root, text="Choisir l'image de Contenu:", bg="#D3D3D3", font=("Arial", 12), fg="black")
content_label.pack(pady=5)

content_path = tk.Entry(root, width=40, font=("Arial", 12))
content_path.pack(pady=5)

content_button = tk.Button(root, text="Charger l'image de contenu", command=lambda: load_and_display_content_image(), font=("Arial", 12), bg="#4CAF50", fg="black")
content_button.pack(pady=5)

style_label = tk.Label(root, text="Choisir l'image de Style:", bg="#D3D3D3", font=("Arial", 12), fg="black")
style_label.pack(pady=5)

style_path = tk.Entry(root, width=40, font=("Arial", 12))
style_path.pack(pady=5)

style_button = tk.Button(root, text="Charger l'image de style", command=lambda: load_and_display_style_image(), font=("Arial", 12), bg="#4CAF50", fg="black")
style_button.pack(pady=5)

# Fonction pour charger l'image de contenu
def load_and_display_content_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        content_path.delete(0, tk.END)
        content_path.insert(0, file_path)
        content_image = load_image(file_path)
        display_image(content_image, content_image_label)

# Fonction pour charger l'image de style
def load_and_display_style_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        style_path.delete(0, tk.END)
        style_path.insert(0, file_path)
        style_image = load_image(file_path)
        display_image(style_image, style_image_label)

# Paramètres pour le transfert de style avec sliders
content_weight_label = tk.Label(root, text="Poids du Contenu:", bg="#D3D3D3", font=("Arial", 12), fg="black")
content_weight_label.pack(pady=5)

content_weight_slider = tk.Scale(root, from_=1e2, to=1e5, orient=tk.HORIZONTAL, resolution=1e2, length=300)
content_weight_slider.set(1e4)
content_weight_slider.pack(pady=5)

style_weight_label = tk.Label(root, text="Poids du Style:", bg="#D3D3D3", font=("Arial", 12), fg="black")
style_weight_label.pack(pady=5)

style_weight_slider = tk.Scale(root, from_=1e-3, to=1e-1, orient=tk.HORIZONTAL, resolution=1e-3, length=300)
style_weight_slider.set(1e-2)
style_weight_slider.pack(pady=5)

epochs_label = tk.Label(root, text="Nombre d'Epoques:", bg="#D3D3D3", font=("Arial", 12), fg="black")
epochs_label.pack(pady=5)

epochs_entry = tk.Entry(root, font=("Arial", 12))
epochs_entry.insert(0, "10")
epochs_entry.pack(pady=5)

# Bouton pour lancer le transfert de style
start_button = tk.Button(root, text="Appliquer le Transfert de Style", command=apply_style_transfer, font=("Arial", 14), bg="#008CBA", fg="black")
start_button.pack(pady=20)

# Lancer l'interface graphique
root.mainloop()
