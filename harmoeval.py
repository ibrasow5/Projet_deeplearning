import tkinter as tk
from tkinter import ttk, messagebox
import csv

# Initialisation de la fenêtre principale
root = tk.Tk()
root.title("Application D'Évaluation des Images")
root.geometry("800x600")
root.configure(bg="#f4f4f4")

# Styles globaux
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12), background="#f4f4f4")
style.configure("TButton", font=("Helvetica", 12), padding=5)
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TCombobox", font=("Helvetica", 12))

# Variables globales
current_image_index = 0
responses = []
images = [
    {"content": "test3.jpg", "style": "style.jpg", "generated": "generated1.jpg"},
    {"content": "content2.jpg", "style": "style2.jpg", "generated": "generated2.jpg"},
    {"content": "content3.jpg", "style": "style3.jpg", "generated": "generated3.jpg"},
]

criteria = ["Fidélité au style", "Respect du contenu", "Esthétique générale", "Originalité"]
ratings = {}

# Fonction pour passer à l'image suivante
def next_image():
    global current_image_index, ratings

    # Vérification des évaluations
    for critere in criteria:
        if ratings[critere].get() == "":
            messagebox.showerror("Erreur", f"Veuillez évaluer le critère : {critere}")
            return

    # Sauvegarder les réponses
    response = {
        "Image": f"Image {current_image_index + 1}",
    }
    for critere in criteria:
        response[critere] = ratings[critere].get()
    responses.append(response)

    # Réinitialiser les notes
    for critere in criteria:
        ratings[critere].set("")

    # Passer à l'image suivante
    current_image_index += 1
    if current_image_index < len(images):
        display_image()
    else:
        save_responses()
        messagebox.showinfo("Terminé", "Merci pour vos évaluations !")
        root.quit()

# Fonction pour afficher une image
def display_image():
    global current_image_index

    # Informations sur l'image actuelle
    image_data = images[current_image_index]

    # Mettre à jour les labels
    lbl_content_image.configure(text=f"Image de Contenu : {image_data['content']}")
    lbl_style_image.configure(text=f"Image de Style : {image_data['style']}")
    lbl_generated_image.configure(text=f"Image Générée : {image_data['generated']}")

    # Mettre à jour le numéro d'image
    lbl_image_number.configure(text=f"Image {current_image_index + 1} sur {len(images)}")

# Fonction pour sauvegarder les réponses dans un fichier CSV
def save_responses():
    with open("evaluation_scores.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Image"] + criteria)
        writer.writeheader()
        writer.writerows(responses)

# Interface utilisateur
frame_top = ttk.Frame(root, padding=20)
frame_top.pack(side=tk.TOP, fill=tk.X)

lbl_title = ttk.Label(frame_top, text="Application d'Évaluation des Images", font=("Helvetica", 16, "bold"))
lbl_title.pack()

frame_images = ttk.Frame(root, padding=20)
frame_images.pack(side=tk.TOP, fill=tk.X)

lbl_image_number = ttk.Label(frame_images, text="Image 1 sur 3", font=("Helvetica", 14))
lbl_image_number.pack()

lbl_content_image = ttk.Label(frame_images, text="Image de Contenu : content1.jpg")
lbl_content_image.pack()

lbl_style_image = ttk.Label(frame_images, text="Image de Style : style1.jpg")
lbl_style_image.pack()

lbl_generated_image = ttk.Label(frame_images, text="Image Générée : generated1.jpg")
lbl_generated_image.pack()

frame_ratings = ttk.Frame(root, padding=20)
frame_ratings.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

ratings = {critere: tk.StringVar() for critere in criteria}

for critere in criteria:
    row = ttk.Frame(frame_ratings, padding=10)
    row.pack(fill=tk.X)

    lbl_critere = ttk.Label(row, text=critere)
    lbl_critere.pack(side=tk.LEFT, padx=5)

    combobox = ttk.Combobox(row, textvariable=ratings[critere], values=["1", "2", "3", "4", "5"], state="readonly")
    combobox.pack(side=tk.RIGHT, padx=5)

frame_bottom = ttk.Frame(root, padding=20)
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

btn_next = ttk.Button(frame_bottom, text="Suivant", command=next_image)
btn_next.pack()

# Afficher la première image
display_image()

# Lancer l'application
root.mainloop()