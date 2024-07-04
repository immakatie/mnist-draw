import tkinter as tk
from tkinter import *
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw

# Загрузка модели
model = tf.keras.models.load_model('mnist_cnn_model.keras')

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition")

        self.canvas = Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack(pady=10)
        
        self.button_predict = Button(self.root, text="Предсказать", command=self.predict_digit)
        self.button_predict.pack(pady=10)

        self.result_label = Label(self.root, text="Предсказание: ")
        self.result_label.pack(pady=10)

        self.button_clear = Button(self.root, text="Очистить", command=self.clear_canvas)
        self.button_clear.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self, img):
        img = img.resize((28, 28))  # Изменение размера до 28x28 пикселей
        img = ImageOps.invert(img)  # Инвертирование цветов
        img = np.array(img).astype('float32') / 255  # Нормализация пикселей
        img = img.reshape(1, 28, 28, 1)  # Изменение формы для подачи в модель
        return img

    def predict_digit(self):
        processed_img = self.preprocess_image(self.image)
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        self.result_label.config(text=f'Предсказание: {predicted_digit}')

if __name__ == "__main__":
    root = Tk()
    app = PaintApp(root)
    root.mainloop()
