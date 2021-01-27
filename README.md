# Tablenet- Deep Learning Model for End-to-end Table Detection and Tabular Data Extraction from Scanned Document Images

# Overview:
A single deep learning model which is capable of detecting the Table inside an image and extract the tabular information. The model will predict the Table and column mask from the input image , based on the generated mask we can filter out the region from the original Image and pytesseract (Tesseract OCR) will be used to extract the Information

# Research Paper:
The details architecture is mentioned in 
https://www.researchgate.net/publication/337242893_TableNet_Deep_Learning_Model_for_End-to-end_Table_Detection_and_Tabular_Data_Extraction_from_Scanned_Document_Images

# Dataset Source:
https://drive.google.com/drive/folders/1QZiv5RKe3xlOBdTzuTVuYRxixemVIODp

# Required Python Libraries
Pillow, Tensorflow, Matplotlib, NumPy, Pandas, xml.etree , Flask , pytesseract

# Description:
Please execute the FINAL.ipynb to check for any sample image. If Table and Column mask are also available then accuracy also can be checked. Directory structure should be as follows:
  /Image_Data: Input image
  /Table_Data: Table-mask  (if available)
  /Column_data: Column-mask (if available)
  
  For end to end execution please execute tablenet-Model_Prediction.ipynb, directory structure should be maintained as mentioned in the runbook
  
  To execute it as web service please execute Deployment/app.py , and then access http:0.0.0.0:8080/upload . a min of 4GB RAM is required to perform the backend operation as i/p image to be converted to high dimensional tensor. address and port number can be updated if requires.
  
# License:
Distributed under the MIT LICENSE

Pull requests are welcome.
