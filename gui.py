# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog,messagebox
from tkinter import *
from PIL import Image,ImageTk
import numpy
import numpy as np


import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import webcolors
import tensorflow as tf
from tensorflow.keras.models import load_model


model  = torch.hub.load('ultralytics/yolov5','yolov5l',pretrained=True)

# Load the model
age_sex_model = load_model('Age_Sex_Detection.keras')

def preprocess_image(image):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the size expected by the model, e.g., 64x64
    image = cv2.resize(image, (48, 48))
    
    # Normalize the image
    image = image / 255.0
    
    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=0)
    
    return image


def predict_sex(image, model = age_sex_model):
    image = preprocess_image(image)
    outputs = model.predict(image)
    sex, age = outputs  # Assuming the output is [age, sex]
    sex = 'Male' if sex <= 0.5 else 'Female'  # Threshold for sex classification
    return sex

def detect_dominant_color(image):
    # Convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert to float
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 1
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the most frequent color
    dominant_color = palette[np.argmax(np.bincount(labels.flatten()))]
    
    return dominant_color

def get_color_category(rgb_color):
    r, g, b = rgb_color
    if r > 70 and g < 100 and b < 100 and r>g and r>b:
        return 'red'
    elif r < 100 and g < 100 and b > 70 and b>g and b>r:
        return 'blue'
    else:
        return 'other'
    
def Detect(img_path):
    img = cv2.imread(img_path)
    results = model(img)
    print(results)
    locs = results.pandas().xyxy[0]
    car_cnt = 0
    male_cnt = 0
    female_cnt = 0
    other_vehicle_no = 0
    for i in range(len(locs)):
        xmin,ymin,xmax,ymax,obj = int(locs['xmin'][i]),int(locs['ymin'][i]),int(locs['xmax'][i]),int(locs['ymax'][i]),locs['name'][i]
        if obj == 'car':
            car_cnt += 1
            color = detect_dominant_color(img[ymin:ymax,xmin:xmax])
            color = get_color_category(color)
            if color == 'blue':
                color = 'red'
            elif color == 'red':
                color = 'blue'
            label = f'{'Car'} {color}'
            cv2.putText(img, label, (int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if obj == 'person':
            person_img = img[ymin:ymax,xmin:xmax]
            sex = predict_sex(person_img)
            if sex == 'Male':
                male_cnt +=1
            else:
                female_cnt +=1
            label = f'{'Person'} {sex}'
            cv2.putText(img, label, (int(xmin), int(ymax)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255, 0, 0),1,1)
    other_vehicle_no = len(locs) - car_cnt - male_cnt - female_cnt
    print('No of Cars: ',car_cnt)
    print('No of Male: ',male_cnt)
    print('No of Female: ',female_cnt)
    print('Other Vehicles: ',other_vehicle_no)
    cv2.imwrite(img_path.split('.')[0]+'new'+'.jpeg',img)
    return car_cnt,male_cnt,female_cnt,other_vehicle_no


top = tk.Tk()
top.geometry('800x600')
top.title('Vehicle Detector')
top.configure(background='#CDCDCD')

# Initializing the labels (1 for Age and 1 for Sex)
label1 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label2 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label3 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label4 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
sign_image = Label(top)

# Detect function

def Detect_Img(file_path):
    global label_packed
    car_cnt,male_cnt,female_cnt,other_vehicle_cnt = Detect(file_path)
    print("No of Cars: ", car_cnt)
    print("No of Male: ", male_cnt)
    print("No of Female: ", female_cnt)
    print("No of Other Vehicles: ", other_vehicle_cnt)
    label1_text = f'Cars Count: {car_cnt}'
    label2_text = f'Male Count: {male_cnt}'
    label3_text = f'Female Count: {female_cnt}'
    label4_text = f'Other Vehicles Count: {other_vehicle_cnt}'
    label1.configure(foreground="#011638",text=label1_text)
    label2.configure(foreground="#011638",text=label2_text)
    label3.configure(foreground="#011638",text=label3_text)
    label4.configure(foreground="#011638",text=label4_text)



# Show Detect Button Function

def show_Detect_button(file_path):
    Detect_b = Button(top,text="Detect Image",command=lambda: Detect_Img(file_path),padx = 10,pady = 5)
    Detect_b.configure(background="#364156",foreground='white',font = ('arial',10,'bold'))
    Detect_b.place(relx = 0.79,rely = 0.46)

# Defining Upload Image Function

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text = '')
        label2.configure(text = '')
        label3.configure(text = '')
        label4.configure(text = '')
        show_Detect_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)
label1.pack(side="bottom",expand=True)
label2.pack(side="bottom",expand=True)
label3.pack(side="bottom",expand=True)
label4.pack(side="bottom",expand=True)
heading=Label(top,text="Vehicle Detection",pady=20,font=('arial',20,"bold"))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()

