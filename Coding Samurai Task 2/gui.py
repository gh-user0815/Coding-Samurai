import tensorflow as tf
import tkinter as tk
from keras.models import load_model
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw

# Load the trained model
model = load_model('model.h5')

class DigitClassifier(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.canvas = tk.Canvas(self, width=300, height=300, bg='white')
        self.canvas.pack()

        self.image = Image.new('L', (300, 300), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_digit)

        self.button = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()



    # def draw_digit(self, event):
    #     x1, y1 = (event.x - 1), (event.y - 1)
    #     x2, y2 = (event.x + 1), (event.y + 1)
    #     self.canvas.create_oval(x1, y1, x2, y2, fill='black')
    #     self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def draw_digit(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_line(x1, y1, x2, y2, fill='black', width=60)
        self.draw.line([x1, y1, x2, y2], fill='black', width=60)

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (300, 300), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        image = self.image.resize((28, 28))
        image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255

        prediction = model.predict(image)
        messagebox.showinfo("Prediction", f"The predicted digit is: {tf.argmax(prediction[0])}")
        # print('Prediction:', (tf.argmax(prediction[0])))

if __name__ == "__main__":
    app = DigitClassifier()
    app.mainloop()