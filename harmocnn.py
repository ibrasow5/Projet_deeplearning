from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour afficher une image
def show_image(image_tensor, title="Image"):
    image = tf.squeeze(image_tensor, axis=0)  # Supprimer la dimension batch
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Charger et prétraiter les images avec Pillow
def load_image_pillow(image_path, target_size=(256, 256)):
    try:
        # Charger l'image avec Pillow
        img = Image.open(image_path)
        img = img.convert("RGB")  # Convertir au format RGB
        img = img.resize(target_size)  # Redimensionner l'image
        
        # Convertir en tableau NumPy, normaliser et convertir en float32
        img_array = np.array(img) / 255.0  # Normaliser entre 0 et 1
        img_array = img_array.astype(np.float32)  # Convertir en float32
        
        # Ajouter une dimension batch
        img_tensor = tf.expand_dims(img_array, axis=0)
        
        return img_tensor
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {image_path}")
        print(f"Détails de l'erreur : {e}")
        return None

# Fonction pour calculer la perte de contenu
def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Fonction pour extraire les caractéristiques de contenu et de style
def get_features(image, model, layers):
    outputs = [model.get_layer(name).output for name in layers]
    feature_model = tf.keras.Model([model.input], outputs)
    features = feature_model(image)
    return features

# Fonction pour calculer la perte de style
def compute_style_loss(base_style, target_style):
    base_style = tf.convert_to_tensor(base_style)
    target_style = tf.convert_to_tensor(target_style)
    gram_base_style = gram_matrix(base_style)
    gram_target_style = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_base_style - gram_target_style))

# Fonction pour calculer la matrice de Gram
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Fonction de transfert de style
def style_transfer(content_image, style_image, content_weight=1e4, style_weight=1e-2, learning_rate=0.02, epochs=100):
    # Charger VGG19
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Définir les couches pour le contenu et le style
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    # Extraire les caractéristiques de contenu et de style
    content_features = get_features(content_image, vgg, content_layers)
    style_features = get_features(style_image, vgg, style_layers)
    
    # Initialiser l'image générée
    generated_image = tf.Variable(content_image, trainable=True, dtype=tf.float32)
    
    # Optimiseur
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Entraînement
    epochs = 30
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            generated_features = get_features(generated_image, vgg, content_layers + style_layers)
            
            content_loss = compute_content_loss(generated_features[0], content_features[0])
            
            style_loss = 0
            for gf, sf in zip(generated_features[1:], style_features):
                style_loss += compute_style_loss(gf, sf)
            style_loss /= len(style_layers)
            
            total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Calculer les gradients
        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        
        # Afficher les pertes toutes les 10 itérations
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}, Content Loss: {content_loss.numpy()}, Style Loss: {style_loss.numpy()}")
    
    return generated_image

# Charger les images source et style
content_image = load_image_pillow("test3.jpg", target_size=(256, 256))
style_image = load_image_pillow("style.png", target_size=(256, 256))

if content_image is None or style_image is None:
    print("Erreur : Impossible de charger l'une des images.")
else:
    # Afficher les images source et style
    show_image(content_image, title="Image de Contenu")
    show_image(style_image, title="Image de Style")
    
    # Appliquer le transfert de style
    result_image = style_transfer(content_image, style_image)
    
    # Afficher l'image générée
    show_image(result_image, title="Image Générée")
