import torch
import urllib
from PIL import Image
import streamlit as st
import os
import os.path as op
from torchvision import models, transforms
st.set_page_config(layout="centered")
st.title("Pak-tAI")

st.write("")
file_up = st.file_uploader("Upload an image", type="jpg")

transform_test = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = models.resnet34()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 54)  
model.load_state_dict(torch.load("Southern.pth", map_location=torch.device('cpu')))
model.eval()

class_names = {
 0:"0",
 1:"1",
 2:"2",
 3:"3",
 4:"4",
 5:"5",
 6:"6",
 7:"7",
 8:"8",
 9:"9",
 10:"10",
 11:"11",
 12:"12",
 13:"13",
 14:"14",
 15:"15",
 16:"16",
 17:"17",
 
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
        if pred_id == 18:
            st.image('nutrition.jpg')
        if pred_id == 18:
            st.image('nutrition.jpg')
        if pred_id == 18:
            st.image('nutrition.jpg')
        if pred_id == 18:
            st.image('nutrition.jpg')            

    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
