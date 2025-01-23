import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                           QComboBox, QSpinBox, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from color_harmonization import ColorHarmonizer
from neural_harmonization import NeuralHarmonizer

class HarmonizationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    
    def __init__(self, harmonizer, image, method):
        super().__init__()
        self.harmonizer = harmonizer
        self.image = image
        self.method = method
    
    def run(self):
        result = self.harmonizer.apply_harmony(self.image, self.method)
        self.finished.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Initialiser les harmonizers
        self.traditional_harmonizer = ColorHarmonizer()
        self.neural_harmonizer = NeuralHarmonizer()
        
        self.current_image = None
        self.harmonized_image = None
    
    def init_ui(self):
        self.setWindowTitle("Harmonisation des Couleurs")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-width: 200px;
            }
            QLabel {
                font-size: 14px;
            }
        """)
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Contrôles supérieurs
        controls_layout = QHBoxLayout()
        
        # Bouton de chargement
        self.load_button = QPushButton("Charger une image")
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)
        
        # Sélection de méthode
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Monochromatique",
            "Complémentaire",
            "Analogue",
            "Triadique",
            "Neural Style Transfer"
        ])
        controls_layout.addWidget(self.method_combo)
        
        # Bouton d'application
        self.apply_button = QPushButton("Appliquer l'harmonisation")
        self.apply_button.clicked.connect(self.apply_harmonization)
        controls_layout.addWidget(self.apply_button)
        
        # Bouton de sauvegarde
        self.save_button = QPushButton("Sauvegarder le résultat")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)
        
        layout.addLayout(controls_layout)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Zone d'affichage des images
        images_layout = QHBoxLayout()
        
        # Image originale
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        images_layout.addWidget(self.original_label)
        
        # Image harmonisée
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        images_layout.addWidget(self.result_label)
        
        layout.addLayout(images_layout)
        
        main_widget.setLayout(layout)
        
        # Définir la taille de la fenêtre
        self.setMinimumSize(900, 600)
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image, self.original_label)
            self.save_button.setEnabled(False)
    
    def apply_harmonization(self):
        if self.current_image is None:
            return
        
        method = self.method_combo.currentText().lower()
        
        if method == "neural style transfer":
            # Pour le neural style, demander l'image de style
            style_file, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner l'image de style",
                "",
                "Images (*.png *.jpg *.jpeg)"
            )
            if style_file:
                style_image = cv2.imread(style_file)
                self.harmonized_image = self.neural_harmonizer.harmonize(
                    self.current_image,
                    style_image
                )
        else:
            # Pour les méthodes traditionnelles
            self.harmonization_thread = HarmonizationThread(
                self.traditional_harmonizer,
                self.current_image,
                method
            )
            self.harmonization_thread.finished.connect(self.harmonization_finished)
            self.harmonization_thread.start()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Mode indéterminé
    
    def harmonization_finished(self, result):
        self.harmonized_image = result
        self.display_image(result, self.result_label)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def save_result(self):
        if self.harmonized_image is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder l'image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            cv2.imwrite(file_name, self.harmonized_image)
    
    def display_image(self, image, label):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        
        # Redimensionner si nécessaire
        max_size = 400
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            w = int(w * scale)
            h = int(h * scale)
            rgb_image = cv2.resize(rgb_image, (w, h))
        
        # Convertir pour Qt
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())