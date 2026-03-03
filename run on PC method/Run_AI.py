import tkinter as tk
from tkinter import filedialog,messagebox
import tensorflow as tf
import numpy as np
import cv2
import requests

ESP_IP = "" #need esp first
stream_url = f"http://{ESP_IP}:81/stream"
predictions = [] #just the list of possible colors must tell the students the order as well.
show_feed=False
frame_height=0
frame_width=0
image=[]

def run_model(folder_path):
    model = tf.keras.models.load_model(folder_path)

    cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        messagebox.showwarning("cannot connect to sorting machine!")
        break
    # Resize to model input size
    b,g,r=cv2.split(frame)
    frame = cv2.merge(r,g,b)

    resized = cv2.resize(frame, (frame_height, frame_width))  # CHANGE to your model input size
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)

    print("Prediction:", class_index)

    requests.get(f"http://{ESP_IP}/cmd?move={prediction}")

def select_file():
    #simple file selection
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        run_model(folder_path)
        show_feed=True

    else:
        messagebox.showwarning("No Selection", "No folder selected!")

#simple UI
root = tk.Tk()
root.title("Teachable Machine")
root.geometry("400x200")

label = tk.Label(root, text="Select a Teachable Machine saved model", pady=20)
label.pack()

select_button = tk.Button(root, text="Select Model", command=select_file, width=20, height=2)
if not show_feed:
    select_button.pack()



root.mainloop()