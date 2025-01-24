import sys
from tkinter import messagebox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QMessageBox
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19

# Fonction pour charger et prétraiter les images
def load_image(image_path, target_size=(256, 256)):
    try:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size  # Sauvegarder la taille originale
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img).unsqueeze(0)  # Ajouter une dimension batch
        return img_tensor, original_size
    except Exception as e:
        return None, None

# Fonction pour calculer la matrice de Gram
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def add_texture(image_tensor, texture_path, texture_weight=0.05):
    texture = Image.open(texture_path).convert("RGB")
    texture = texture.resize((image_tensor.size(3), image_tensor.size(2)))

    texture_tensor = transforms.ToTensor()(texture).unsqueeze(0)
    
    return (image_tensor + texture_tensor * texture_weight).clamp(0, 1)  

# Classe pour extraire les caractéristiques avec VGG19
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.style_layers = [0, 5, 10, 19, 28]
        self.content_layer = 21
        self.layers = nn.Sequential(*list(vgg.children())[:29])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        style_features = []
        content_feature = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.style_layers:
                style_features.append(x)
            if i == self.content_layer:
                content_feature = x
        return content_feature, style_features

# Fonction de transfert de style
def style_transfer(content_image, style_image, content_weight=1e1, style_weight=1e5, learning_rate=0.01, epochs=500):
    model = VGGFeatures().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    generated = content_image.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([generated], lr=learning_rate)

    content_feature, _ = model(content_image)
    _, style_features = model(style_image)
    style_grams = [gram_matrix(sf) for sf in style_features]

    for epoch in range(epochs):
        optimizer.zero_grad()
        gen_content_feature, gen_style_features = model(generated)
        content_loss = torch.mean((gen_content_feature - content_feature) ** 2)

        style_loss = 0
        for gf, sg in zip(gen_style_features, style_grams):
            gm = gram_matrix(gf)
            style_loss += torch.mean((gm - sg) ** 2)

        style_loss /= len(style_features)
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss.item()}")

    return generated.detach()

# Application principale avec PyQt5
class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Application de Transfert de Style")
        self.setGeometry(100, 130, 800, 500)

        self.content_image_tensor = None
        self.style_image_tensor = None
        self.result_image_tensor = None
        self.original_content_size = None  # Taille originale de l'image de contenu

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        content_layout = QVBoxLayout()
        self.content_label = QLabel("Image de Contenu", self)
        self.content_label.setAlignment(Qt.AlignCenter)
        self.content_frame = self.create_image_frame("Aucune image chargée")
        load_content_btn = QPushButton("Charger Image de Contenu", self)
        load_content_btn.clicked.connect(self.load_content_image)
        content_layout.addWidget(self.content_label)
        content_layout.addWidget(self.content_frame, alignment=Qt.AlignCenter)
        content_layout.addWidget(load_content_btn, alignment=Qt.AlignCenter)

        style_layout = QVBoxLayout()
        self.style_label = QLabel("Image de Style", self)
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_frame = self.create_image_frame("Aucune image chargée")
        load_style_btn = QPushButton("Charger Image de Style", self)
        load_style_btn.clicked.connect(self.load_style_image)
        style_layout.addWidget(self.style_label)
        style_layout.addWidget(self.style_frame, alignment=Qt.AlignCenter)
        style_layout.addWidget(load_style_btn, alignment=Qt.AlignCenter)

        result_layout = QVBoxLayout()
        self.result_label = QLabel("Résultat", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_frame = self.create_image_frame("Résultat après transfert")
        apply_btn = QPushButton("Appliquer Transfert de Style", self)
        apply_btn.clicked.connect(self.apply_style_transfer)
        save_btn = QPushButton("Enregistrer Résultat", self)
        save_btn.clicked.connect(self.save_result_image)
        result_layout.setContentsMargins(0, 38, 0, 0)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_frame, alignment=Qt.AlignCenter)
        result_layout.addWidget(apply_btn, alignment=Qt.AlignCenter)
        result_layout.addWidget(save_btn, alignment=Qt.AlignCenter)

        layout.addLayout(content_layout)
        layout.addLayout(style_layout)
        layout.addLayout(result_layout)

        self.setLayout(layout)

    def create_image_frame(self, text):
        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setFixedSize(400, 400)
        label = QLabel(text, frame)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font: 16pt Arial; color: gray;")
        label.setGeometry(0, 0, 400, 400)
        return frame

    def load_content_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir Image de Contenu", "", "Images (*.jpg *.jpeg *.png *.gif)")
        if path:
            self.content_image_tensor, self.original_content_size = load_image(path)
            if self.content_image_tensor is not None:
                self.display_image(self.content_frame, self.content_image_tensor)

    def load_style_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir Image de Style", "", "Images (*.jpg *.jpeg *.png *.gif)")
        if path:
            self.style_image_tensor, _ = load_image(path)
            if self.style_image_tensor is not None:
                self.display_image(self.style_frame, self.style_image_tensor)

    def display_image(self, frame, img_tensor):
        img_tensor = img_tensor.squeeze(0).cpu().clone()
        img_tensor = img_tensor.permute(1, 2, 0).numpy()
        img_tensor = img_tensor * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_tensor = (img_tensor * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(img_tensor)
        img = img.convert("RGB")
        img = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap(img)
        label = QLabel(self)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        frame.setLayout(QVBoxLayout())
        frame.layout().addWidget(label, alignment=Qt.AlignCenter)

    def apply_style_transfer(self):
        if self.content_image_tensor is None or self.style_image_tensor is None:
            QMessageBox.warning(self, "Erreur", "Veuillez charger une image de contenu et une image de style.")
            return

        QMessageBox.information(self, "Transfert en Cours", "Le transfert de style est en cours. Cela peut prendre un moment.")
        QTimer.singleShot(0, self.start_style_transfer)

    def start_style_transfer(self):
        try:
            self.result_image_tensor = style_transfer(self.content_image_tensor, self.style_image_tensor, epochs=10)
            self.result_image_tensor = add_texture(self.result_image_tensor, "style.png")
            self.display_image(self.result_frame, self.result_image_tensor)
            QMessageBox.information(self, "Transfert Terminé", "Le transfert de style est terminé.")
        except Exception as e:
            print(f"Erreur lors du transfert de style : {e}")

    def save_result_image(self):
        if self.result_image_tensor is None:
            QMessageBox.warning(self, "Erreur", "Aucun résultat à sauvegarder.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img_tensor = self.result_image_tensor.squeeze(0).cpu().clone()
            img_tensor = img_tensor.permute(1, 2, 0).numpy()
            img_tensor = img_tensor * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_tensor = (img_tensor * 255).clip(0, 255).astype("uint8")
            img = Image.fromarray(img_tensor)
            img = img.resize(self.original_content_size)  # Redimensionner à la taille d'origine
            img.save(path)
            QMessageBox.information(self, "Sauvegarde", f"Image sauvegardée sous : {path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())
