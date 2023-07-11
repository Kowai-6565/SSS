import torch
import urllib
from PIL import Image
import streamlit as st
import os
import os.path as op
from torchvision import models, transforms
st.set_page_config(layout="centered")
st.title("Pak-tAI")
st.image('Nutrition/0.PNG')

st.write("")
file_up = st.file_uploader("Upload an image", type="jpg")

transform_test = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = models.resnet34()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 18)  
model.load_state_dict(torch.load("Southern.pth", map_location=torch.device('cpu')))
model.eval()

class_names = {
 0:"ไก่ต้มขมิ้น",
 1:"เเกงคั่วหอยขมใบชะพลู",
 2:"ขนมจีนนํ้ายาปู",
 3:"ไก่กอเเละ",
 4:"ไก่ทอดหาดใหญ่",
 5:"ไข่ครอบ",
 6:"ข้าวยํา",
 7:"คั่วกลิ้งหมู",
 8:"ใบเหลียงต้มกะทิกุ้งสด",
 9:"หมูฮ้อง",
 10:"นํ้าพริกกะปิ",
 11:"นํ้าพริกกุ้งเสียบ",
 12:"ปลากรายทอดขมิ้น",
 13:"เเกงเหลืองปลากระพง",
 14:"ใบเหลียงผัดไข่",
 15:"หมูผัดกะปิ",
 16:"สะตอผัดกุ้ง",
 17:"เเกงไตปลา",
 
}

model.food_class = class_names 

if file_up is not None:

        
    temp_file_path = "temp.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(file_up.getvalue())

    img = Image.open(temp_file_path)

    st.image(img, caption='รูปอาหาร', use_column_width=True)

    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(torch_images)

        _, predict = torch.max(outputs, 1)
        pred_id = predict.item()
        st.write('ชนิดอาหาร:', model.food_class[pred_id])
        st.write(pred_id)
        '''if pred_id == 0:
            st.image('Nutrition/0.PNG')'''

        '''if pred_id == 1:
            st.image('Nutrition/1.PNG')

        if pred_id == 2:
            st.image('Nutrition/2.PNG')

        if pred_id == 3:
            st.image('Nutrition/3.PNG')

        if pred_id == 4:
            st.image('Nutrition/4.PNG')

        if pred_id == 5:
            st.image('Nutrition/5.PNG')

        if pred_id == 6:
            st.image('Nutrition/6.PNG')

        if pred_id == 7:
            st.image('Nutrition/7.PNG')

        if pred_id == 8:
            st.image('Nutrition/8.PNG')

        if pred_id == 9:
            st.image('Nutrition/9.PNG')

        if pred_id == 10:
            st.image('Nutrition/10.PNG')

        if pred_id == 11:
            st.image('Nutrition/11.PNG')

        if pred_id == 12:
            st.image('Nutrition/12.PNG')

        if pred_id == 13:
            st.image('Nutrition/13.PNG')

        if pred_id == 14:
            st.image('Nutrition/14.PNG')

        if pred_id == 15:
            st.image('Nutrition/15.PNG')

        if pred_id == 16:
            st.image('Nutrition/16.PNG')

        if pred_id == 17:
            st.image('Nutrition/17.PNG')  '''

    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
