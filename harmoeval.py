import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import csv

# Initialisation de la fenêtre principale
root = tk.Tk()
root.title("Application D'Évaluation des Images")
root.geometry("900x650")
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
    {"content": "test3.jpg", "style": "style.png", "generated": "generated1.jpg"},
    {"content": "t1.jpg", "style": "style.png", "generated": "generated2.jpg"},
    {"content": "t2.jpg", "style": "style.png", "generated": "generated3.jpg"},
]

criteria = ["Fidélité au style", "Respect du contenu", "Esthétique générale", "Originalité"]
ratings = {}
image_widgets = {}

# Fonction pour passer à l'image suivante
def next_image():
    global current_image_index, ratings

    # Vérification des évaluations
    for critere in criteria:
        if ratings[critere].get() == "":
            messagebox.showerror("Erreur", f"Veuillez évaluer le critère : {critere}")
            return

    # Calculer la moyenne des évaluations pour l'image actuelle
    total_score = 0
    for critere in criteria:
        total_score += int(ratings[critere].get())
    average_score = total_score / len(criteria)

    # Sauvegarder les réponses
    response = {
        "Image": f"Image {current_image_index + 1}",
    }
    for critere in criteria:
        response[critere] = ratings[critere].get()
    response["Moyenne"] = round(average_score, 2)  # Ajouter la moyenne à la réponse
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

    try:
        # Charger les images
        content_image = Image.open(image_data['content']).resize((240, 150))
        style_image = Image.open(image_data['style']).resize((240, 150))
        generated_image = Image.open(image_data['generated']).resize((240, 150))

        # Convertir en images Tkinter
        content_photo = ImageTk.PhotoImage(content_image)
        style_photo = ImageTk.PhotoImage(style_image)
        generated_photo = ImageTk.PhotoImage(generated_image)

        # Mettre à jour les widgets
        image_widgets['content'].configure(image=content_photo)
        image_widgets['content'].image = content_photo  # Nécessaire pour éviter que l'image soit supprimée par le garbage collector

        image_widgets['style'].configure(image=style_photo)
        image_widgets['style'].image = style_photo

        image_widgets['generated'].configure(image=generated_photo)
        image_widgets['generated'].image = generated_photo

        # Mettre à jour le numéro d'image
        lbl_image_number.configure(text=f"Image {current_image_index + 1} sur {len(images)}")
    except FileNotFoundError as e:
        messagebox.showerror("Erreur", f"Fichier non trouvé : {e}")

# Fonction pour sauvegarder les réponses dans un fichier CSV
def save_responses():
    with open("evaluation_scores.csv", "w", newline="", encoding="utf-8") as file:
        fieldnames = ["Image"] + criteria + ["Moyenne"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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

# Cadre pour afficher les images horizontalement
frame_image_display = ttk.Frame(frame_images)
frame_image_display.pack(side=tk.TOP, fill=tk.X)

image_widgets['content'] = ttk.Label(frame_image_display, text="Image de Contenu", relief="solid", padding=10)
image_widgets['content'].pack(side=tk.LEFT, padx=10)

image_widgets['style'] = ttk.Label(frame_image_display, text="Image de Style", relief="solid", padding=10)
image_widgets['style'].pack(side=tk.LEFT, padx=10)

image_widgets['generated'] = ttk.Label(frame_image_display, text="Image Générée", relief="solid", padding=10)
image_widgets['generated'].pack(side=tk.LEFT, padx=10)

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
