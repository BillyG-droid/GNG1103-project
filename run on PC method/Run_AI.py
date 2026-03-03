import tkinter as tk
from tkinter import filedialog,messagebox
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image, ImageTk

ESP_IP = "" #need esp first
stream_url = f"http://{ESP_IP}:81/stream"
classes = [] #just the list of possible colors must tell the students the order as well.
show_feed=False
frame_height=0
frame_width=0

def run_model(folder_path):
    model = tf.keras.models.load_model(folder_path)

    cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        messagebox.showwarning("cannot connect to sorting machine!")
        break
    # Resize to model input size

    resized = cv2.resize(frame, (frame_height, frame_width))  # CHANGE to our model input size
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    #returns an array of probabilities for every possible output i.e. if three outputs something like ([0.05,0.9,0.01])
    prediction = model.predict(input_data)
    #returns the index with the largest probability
    class_index = np.argmax(prediction)
    #returns the highest probability
    confidence = float(np.max(prediction))

    #via wifi tells the esp what the prediction was
    requests.get(f"http://{ESP_IP}/cmd?prediction={class_index}")

    text = f"{classes[class_index]} confidence:({confidence:.2f})"
    #creates a rectangle with th image in it
    # paramaters: image, top left, bottom right, Rectangle color,thichness( negative means filled rect)
    cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
    #puts text with image
    # parameters: image,text, bottom left,font,font size mulitipier (each font has a font size so must apply multiplier to change size),color,thickness of lines to draw text 
    cv2.putText(frame, text,(20, 45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    # cv2 stores images in BGR so must convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to Tkinter image
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(30, update_frame)


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
root.title("AI hopper sorter")
root.geometry("400x200")

label = tk.Label(root, text="Select a Teachable Machine saved model", pady=20)
label.pack()

select_button = tk.Button(root, text="Select Model", command=select_file, width=20, height=2)
if not show_feed:
    select_button.pack()

root.mainloop()