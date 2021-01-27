
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D

import datetime

from PIL import Image
import statistics
import pytesseract

from flask import Flask, render_template, request,jsonify

UPLOAD_FOLDER = 'uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# %% [code]
image_height=1024
image_width=1024



# %% [code]
def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    #input_mask -= 1
    return input_image

# %% [code]
def decode_image(image):
    img=tf.io.decode_jpeg(image)
    img=tf.image.resize(img, [image_height, image_width])
    return img

# %% [code]
def decode_mask(image):
    img=tf.io.decode_jpeg(image,channels=1)
    img=tf.image.resize(img, [image_height, image_width])
    return img

# %% [code]
def process_1(file_paths):
    img = normalize(decode_image(tf.io.read_file(file_paths)))
    return img

# %% [code]
def process_2(file_paths):
    img = normalize(decode_image(tf.io.read_file(file_paths)))
    
    mask_path=tf.strings.regex_replace(file_paths,'.jpg','.jpeg')
    
    tab_mask=tf.strings.regex_replace(mask_path,"Image_Data", "Table_Data")
    col_mask=tf.strings.regex_replace(mask_path,"Image_Data", "Column_Data")
    
    table_mask = normalize(decode_mask(tf.io.read_file(tab_mask)))
    column_mask=normalize(decode_mask(tf.io.read_file(col_mask)))
    
    return img, {'table_mask':table_mask,'column_mask':column_mask}

# %% [code]
def create_mask(pred_mask1, pred_mask2):
    pred_mask1 = tf.argmax(pred_mask1, axis=-1)
    pred_mask1 = pred_mask1[..., tf.newaxis]
    
    pred_mask2 = tf.argmax(pred_mask2, axis=-1)
    pred_mask2 = pred_mask2[..., tf.newaxis]
    return pred_mask1[0], pred_mask2[0]

# %% [code]
def show_prediction_sample_image(dataset=None, num=1):
    
    model = tf.keras.models.load_model('mymodel_45')
    
    
    for image in dataset.take(num):
        pred_mask1, pred_mask2 = model.predict(image, verbose=1)
        table_mask, column_mask = create_mask(pred_mask1, pred_mask2)
        
        img1=tf.keras.preprocessing.image.array_to_img(image[0])
        im.save('image.bmp')
        
        img2=tf.keras.preprocessing.image.array_to_img(table_mask)
        im.save('table_mask.bmp')
        
        img3=tf.keras.preprocessing.image.array_to_img(column_mask)
        im.save('column_mask.bmp')
        
    
    return 'img1','img2','img3' 
		

# %% [code]
def generate_segment(img1,img2,img3):
    #img_org  = Image.open('./image.bmp')
    img_org=img1
	#img_mask = Image.open('./table_mask.bmp')
    img_mask=img2
    img_mask = img_mask.convert('L')
    img=img_org.putalpha(img_mask)
    return img 
    #img_org.save('output.png')

# %% [code]
def ocr_core(filename):
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

# %% [code]
def get_mask(dataset=None, num=1):
    
    table=[]
    column=[]
    for i in dataset:
        table.append(i[1]['table_mask'])
        column.append(i[1]['column_mask'])
    
    model = tf.keras.models.load_model('../input/model50/all/mymodel_45')
    
    pred_tab=[]
    pred_col=[]
    for image, (mask1, mask2) in dataset.take(num):
        pred_mask1, pred_mask2 = model.predict(image, verbose=1)
        table_mask, column_mask = create_mask(pred_mask1, pred_mask2)
        pred_tab.append(table_mask)
        pred_col.append(column_mask)
            
    return table,column,pred_tab,pred_col


    


@app.route('/upload')
def upload_files():
   return render_template('upload.html')

   
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

   if request.method == 'POST':
      f = request.files['file']
      name=f.filename
      #location='uploads/' + name
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
      #f.save(secure_filename(location))
      img_path='uploads/*.jpg'
      list_ds = tf.data.Dataset.list_files(img_path)
      DATASET_SIZE = len(list(list_ds))
      test_size = DATASET_SIZE
      test = list_ds.take(test_size)
      BATCH_SIZE = 1
      BUFFER_SIZE = 1000
      test = test.map(process_1)
      test_dataset = test.batch(BATCH_SIZE)
      img1,img2,img3=show_prediction_sample_image(test_dataset)
      #img=generate_segment()
      #text=ocr_core(img)
	  
      
      #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], name))
      return jsonify({'Table_data': 'text'})
		
if __name__ == '__main__':
   app.run(debug = True)
