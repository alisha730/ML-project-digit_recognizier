# app.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

# --------------------- WINDOW ---------------------
root = tk.Tk()
root.title("✨ Unique Digit Recognizer ✨")
root.geometry("350x450")
root.resizable(False, False)
root.configure(bg="#2c3e50")  # dark blue background

# --------------------- TITLE FRAME ---------------------
title_frame = tk.Frame(root, bg="#34495e", height=50)
title_frame.pack(fill="x")
title_label = tk.Label(title_frame, text="Draw a Digit", font=("Arial", 18, "bold"), bg="#34495e", fg="white")
title_label.pack(pady=10)

# --------------------- CANVAS ---------------------
canvas_width, canvas_height = 250, 250
canvas_frame = tk.Frame(root, bg="#2c3e50")
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg="white", bd=5, relief="ridge")
canvas.pack()

# For drawing
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x-12), (event.y-12)
    x2, y2 = (event.x+12), (event.y+12)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# --------------------- PREDICTION LABEL ---------------------
result_label = tk.Label(root, text="Prediction: None", font=("Arial", 20, "bold"), bg="#2c3e50", fg="#e74c3c")
result_label.pack(pady=15)

# --------------------- FUNCTIONS ---------------------
def predict_digit():
    # Resize to 28x28
    img_resized = image.resize((28,28))
    img_array = np.array(img_resized)
    
    # Invert and normalize
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1,28,28)
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    
    result_label.config(text=f"Prediction: {digit}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height], fill=255)
    result_label.config(text="Prediction: None")

# --------------------- BUTTONS ---------------------
button_frame = tk.Frame(root, bg="#2c3e50")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="Predict", font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
                           activebackground="#2ecc71", width=10, command=predict_digit)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", font=("Arial", 14, "bold"), bg="#c0392b", fg="white",
                         activebackground="#e74c3c", width=10, command=clear_canvas)
clear_button.grid(row=0, column=1, padx=10)

# --------------------- INSTRUCTIONS ---------------------
instructions = tk.Label(root, text="Draw a digit 0-9 in the box above.\nClick Predict to see the result.", 
                        font=("Arial", 10), bg="#2c3e50", fg="white")
instructions.pack(pady=5)

# --------------------- RUN APP ---------------------
root.mainloop()
