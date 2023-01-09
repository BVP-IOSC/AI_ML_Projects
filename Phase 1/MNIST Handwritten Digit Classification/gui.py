from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import PhotoImage
import numpy as np
import cv2
from sklearn.svm import SVC
import pandas as pd
import os
from matplotlib.image import imread

traindf = pd.read_csv(r'D:/python/ml/MNIST-Digit-Classification/mnist_train.csv')
x_train = traindf.drop('label',axis=1)
y_train = traindf['label']
svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(x_train, y_train)

root = tk.Tk()
root.configure(bg="#444444")
icon = PhotoImage(file = 'ml/MNIST-Digit-Classification/icon.png')
root.iconphoto(False, icon)
root.resizable('0', '0')
root.title("Digit Identifier")

canvas = tk.Canvas(root, bg='black', height=300, width=300)
canvas.grid(row=0, column=0, columnspan=4)
img = Image.new('RGB', (300,300), ('black'))
imagedraw = ImageDraw.Draw(img)
count = 0
text_change = "Draw me a number!"
def draw(event):
    x , y = event.x , event.y
    x1 , y1 = x-10 , y-10
    x2, y2 = x+10 , y+10

    canvas.create_oval((x1, y1,x2,y2), fill='white', outline='white')
    imagedraw.ellipse((x1, y1,x2,y2), fill='white', outline='white')

def clear_canvas():
    global img, imagedraw
    img = Image.new('RGB', (300,300), ('black'))
    imagedraw = ImageDraw.Draw(img)
    canvas.delete('all')
    

def predict():
    global count
    global text_change
    global pred_svm_rbf
    imagearray = np.array(img)
    imagearray = cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(imagearray, (28,28))
    cv2.imwrite(str(count)+'.jpg',image1)
    read_img = imread(str(count)+'.jpg')
    convert = np.array(read_img).reshape(-1,784)  
    for i in range(0,count):
        os.remove(str(count+i)+'.jpg')
        count = count+1
    
    pred_svm_rbf = svm_rbf.predict(convert)
    text_change = "I think it is . . ." + str(pred_svm_rbf[0])
    label.config(text = text_change)
    

canvas.bind("<B1-Motion>", draw)

button_predict = tk.Button(root, text='PREDICT', width=10, height=2, bg='black', fg='white', font='Helvetica', command= predict)
button_predict.grid(row=2,column=0)

button_clear = tk.Button(root, text='CLEAR', width=10, height=2, bg='black', fg='white', font='Helvetica', command= clear_canvas)
button_clear.grid(row=2,column=2)

button_exit= tk.Button(root, text='EXIT', width=10, height=2, bg='black', fg='white', font='Helvetica', command= root.destroy)
button_exit.grid(row=2,column=3)

label = tk.Label(root, text=text_change, bg = 'black', fg='white', font = 'Helvetica', width=33, height=2)
label.grid(row=3, column=0, columnspan=4)

root.mainloop()
