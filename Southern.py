import torch
import urllib
from PIL import Image
import streamlit as st
import os
import os.path as op
from torchvision import models, transforms


st.set_page_config(layout="centered")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.title(" Puk-tAI  ")    
with col3:
    st.write(' ')

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.title("ปักษ์ใต้")    
with col3:
    st.write(' ')    


st.write(" AI คัดเเยกรูปภาพอาหารใต้ พร้อมบอกค่าโภชนาการ รสชาติ และส่วนประกอบของอาหาร ")    

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
        
        ## เเกงจืดต้มขมิ้น ##
        if pred_id == 0:
            st.image('Nutrition/0.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    
               
        ## เเกงคั่วหอยขม ##
        if pred_id == 1:
            st.image('Nutrition/1.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/02/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ตั้งกระทะผัดพริกเเกง ")
                st.write("ตั้งกระทะโดยใช้ไฟกลาง พอกระทะร้อนใส่นํ้ามันลงไป ตามด้วยพริกเเกง ผัดให้เข้ากันจนหอม")
                st.write("นําหัวกะทิใส่ลงไป อย่าใส่จนหมดนะครับ เก็บหางกะทิไว้ เเล้วผัดจนกะทิเเตกมัน")
                st.write("เมื่อกะทิเเตกมันได้เเล้ว เติมหางกะทิเลยครับ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/02/2.JPG', caption="นําพริกเเกงลงไปผัด", use_column_width=True)
                with col2:
                    st.image('Recipe/02/3.JPG', caption="เติมหางกะทิลงไป", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่ส่วนผสม + ปรุงรส ")
                st.write("เมื่อกะทิเดือดได้ที่เเล้ว นําหอยขมใส่ลงไป ตามด้วยใบชะพลู ชะอม เเละเเครอท คนให้ส่วนผสมเข้ากันดี ทิ้งไว้สักพักนึงครับ")
                st.write("เติมหางกะทิลงไปรอบ ๆ เพื่อช่วยให้นํ้าเเกงไม่เเห้งครับ ปิดผฝาทิ้งไว้")
                st.write("เมื่อส่วนผสมเข้ากันดีเเล้ว ทําการปรุงรสด้วย นํ้าปลา นํ้าตาลปี้บ คนให้เข้ากันอีกครั้ง เมื่อได้รสชาติที่ต้องการเเล้ว ใส่ใบมะกรูดลงไปครับ จากนั้นคนให้เข้ากัน เเล้วปิดไฟเลยครับ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/02/4.JPG', caption="ใส่ใบชะพลูลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/02/5.JPG', caption="ใส่ชะอมลงไป", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/02/6.JPG', caption="ใส่เเครอทลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/02/7.JPG', caption="ปรุงรสด้วยนํ้าปลา", use_column_width=True)
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("ตักเเกงคั่วหอยขมขึ้นมาใส่ชาม โรยด้วยใบมะกรูดหั่นฝอย ตกเเต่งให้สวยงาม เเล้วก็จัดเสิร์ฟได้เลยครับ")
                st.image('Recipe/02/8.JPG', caption="เมนู เเกงคั่วหอยขม พร้อมเสิร์ฟเเล้วครับ!", use_column_width=True)

        ## ขนมจีนนํ้ายากะทิ ##
        if pred_id == 2:
            st.image('Nutrition/2.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/03/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เตรียมเครื่องแกงน้ำยากะทิ ")
                st.write("ต้มน้ำให้เดือด ใส่ตะไคร้ กระเทียม หอมแดง ข่า กระชาย และ พริกแห้งลงไป ตามด้วยเนื้อปลา ต้มจนสุกกรองทุกอย่างออก และเก็บน้ำต้มไว้ด้วย")
                st.write("นําผักเเละเนื้อปลาที่ต้ม มาโขลกให้ละเอียดจนกลายเป็นเครื่องเเกง")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/2.JPG', caption="ใส่ตะไคร้ กระเทียม หอมเเดง ข่า กระชาย เเละพริกเเห้งลงไป เเละตามด้วยเนื้อปลา ต้มจนสุก", use_column_width=True)
                with col2:
                    st.image('Recipe/03/3.JPG', caption="นําผัก เเละเนื้อปลาที่ต้มมาโขลกให้ละเอียด", use_column_width=True)    
                
                st.title("STEP 2 : ทํานํ้ายากะทิ ")
                st.write("ตั้งกระทะเทน้ำกะทิลงไปครึ่งหนึ่ง รอจนร้อนได้ที่ ให้นำเครื่องแกงลงไปผัดกับกะทิจนเข้ากันแล้ว ให้เติมกะทิส่วนที่เหลือลงไป ตามด้วยน้ำเปล่าเคี่ยวจนกะทิแตกมัน")
                st.write("ปรุงรสด้วยน้ำปลา เกลือ และตามด้วยใส่ลูกชิ้นปลาลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/4.JPG', caption="นำเครื่องแกงลงไปผัดกับกะทิจนเข้ากัน", use_column_width=True)
                with col2:
                    st.image('Recipe/03/5.JPG', caption="เติมกะทิส่วนที่เหลือลงไปในเครื่องแกงที่ผัดไว้ก่อนหน้านี้", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("ตักน้ำยากะทิราดบนขนมจีน เสิร์ฟคู่กับผักแกล้ม เพียงเท่านี้ “ขนมจีนน้ำยากะทิ” ของเราก็พร้อมรับประทานแล้ว")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/6.JPG', caption="ตักน้ำยากะทิใส่จานเตรียมเสิร์ฟ", use_column_width=True)
                with col2:
                    st.image('Recipe/03/7.JPG', caption="ขนมจีนนํ้ายากะทิ พร้อมรับประทานเเล้วครับ", use_column_width=True)    
        
        ## ไก่กอเเละ ##
        if pred_id == 3:
            st.image('Nutrition/3.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/04/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ทําพริกเเกง ")
                st.write("เริ่มทำพริกแกง โดยการใส่พริกชี้ฟ้าแห้งแช่น้ำสับ พริกขี้หนูแห้งแช่น้ำสับ หอมแดง กระเทียม ลูกผักชี ยี่หร่า อบเชย และกะปิ ลงในครก แล้วโขลกให้ละเอียดเข้ากัน")
                st.write("ตั้งกระทะใส่น้ำมันพืชจนพอร้อน นำเครื่องแกงที่โขลกไว้ลงไปผัดจนหอม ใส่กะทิลงไปเคี่ยวจนแตกมัน")
                st.write("ปรุงรสด้วยน้ำมะขามเปียก น้ำตาลปี๊บ และน้ำปลา เคี่ยวต่อซักครู่แล้วยกลงพักไว้ให้เย็น")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/2.JPG', caption="โขลกพริกแกงให้ละเอียด", use_column_width=True)
                with col2:
                    st.image('Recipe/04/3.JPG', caption="ใส่กะทิลงไปเคี่ยวจนแตกมัน", use_column_width=True)    
                
                st.title("STEP 2 : หมักไก่ ")
                st.write("โขลกกระเทียม ผงขมิ้น และเกลือป่น ให้ละเอียด")
                st.write("หมักสะโพกไก่หั่นชิ้น ด้วยส่วนผสมที่โขลกไว้ และกะทิ คลุกเคล้าให้เข้ากัน หมักไว้ 30 นาที")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/4.JPG', caption="โขลกกระเทียม ผงขมิ้น และเกลือป่น ให้ละเอียด", use_column_width=True)
                with col2:
                    st.image('Recipe/04/5.JPG', caption="หมักไก่ไว้ 30 นาที", use_column_width=True)
                
                st.title("STEP  3: ย่าง ")
                st.write("นำไก่ที่หมักไว้มาเสียบไม้ แล้วนำขึ้นเตาย่าง คอยทาด้วยพริกแกงที่ทำไว้อย่างสม่ำเสมอทั้งสองด้าน ย่างจนสุกสวย จึงนำขึ้นจัดเสิร์ฟ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/6.JPG', caption="นำไก่ที่หมักไว้มาเสียบไม้", use_column_width=True)
                with col2:
                    st.image('Recipe/04/7.JPG', caption="ทาด้วยพริกแกงที่ทำไว้อย่างสม่ำเสมอทั้งสองด้าน", use_column_width=True)    
        
        ## ไก่ทอด ##
        if pred_id == 4:
            st.image('Nutrition/4.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/05/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
                st.write("1.ล้างไก่ให้สะอาด พักในตะแกรงให้สะเด็ดน้ำ")
                st.write("2.โขลก พริกไทย, ลูกผักชี, และยี่หร่า เข้าด้วยกันให้ละเอียด")
                st.write("3.เสร็จแล้วค่อยใส่กระเทียม,เกลือป่น และน้ำตาลทรายแดง โขลกทุกอย่างให้ละเอียดอีกครั้ง")
                st.write("4.จากนั้นนำทุกอย่างไปผสมกับไก่ และ ซีอิ๊วขาวหรือซอสปรุงรส คลุกส่วนผสม ทั้งหมดให้เข้ากัน หมักทิ้งในตู้เย็น อย่างน้อย 3 ชั่วโมง")
                st.write("5.เมื่อหมักไก่จนได้ที่แล้ว ให้นำไปคลุกกับแป้งสาลีและ แป้งข้าวเจ้า")
                st.write("6.นำไก่ทอดในน้ำมันพืชที่ร้อนได้ที่ด้วยไฟปานกลาง จนสุกเป็นสีน้ำตาลทอง (ใช้เวลาประมาณ 10-12 นาที) ตักขึ้น พักในบนกระดาษซับมันประมาณ 3-4 นาที")
                st.write("7.เสิร์ฟคู่กับหอมเจียว และ น้ำจิ้มไก่ตราแม่ประนอมก็ดีงามใช่ย่อย")
                
        ## ไข่ครอบ ##
        if pred_id == 5:
            st.image('Nutrition/5.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/06/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เเยกไข่ ")
                st.write("นำไข่ออกจากเปลือก อย่าให้เปลือกแตกหมดนะครับ เพราะต้องใช้เปลือกไข่นำไปนึ่ง แยกไข่แดงกับไข่ขาว ค่อย ๆ แยกนะคะ ไม่งั้นไข่แดงจะแตก แล้วนำไข่แดงแช่น้ำเกลือ 5 ชั่วโมง")
                st.write("Tip.. เกลือจะช่วยให้ไข่เป็นก้อนไม่เหลวครับ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/2.JPG', caption="เลาะเปลือกไข่ข้างบนออก ใส่ไข่ลงไปในภาชนะ", use_column_width=True)
                with col2:
                    st.image('Recipe/06/3.JPG', caption="ใส่เกลือลงไปในน้ำ", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/4.JPG', caption="แยกไข่แดงออกจากไข่ขาว นำไปแช่น้ำเกลือ", use_column_width=True)
                with col2:
                    st.image('Recipe/06/5.JPG', caption="แช่น้ำเกลือ 5 ชั่วโมง", use_column_width=True)        
                
                st.title("STEP 2 : ตัดแต่งเปลือกและหยอดไข่ ")
                st.write("นำเปลือกไข่ที่เราเอาไข่ออกมาตัดโดยใช้กรรไกร ซึ่งเราจะตัดเหลือไว้ครึ่งลูก เพื่อใส่ไข่ไว้ข้างใน ")
                st.write("เมื่อพอครบ 5 ชั่วโมงแล้ว นำไข่แดงมาใส่เปลือก โดยใส่ไข่ 2 ฟองลงไปในเปลือกไข่ 1 เปลือกครับ ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/6.JPG', caption="ตัดแต่งเปลือกไข่ ", use_column_width=True)
                with col2:
                    st.image('Recipe/06/7.JPG', caption="ตัดแต่งเปลือกไข่ ", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/8.JPG', caption="นำไข่แดงใส่ในเปลือกไข่ที่เราตัดไว้", use_column_width=True)
                with col2:
                    st.image('Recipe/06/9.JPG', caption="ใส่เรียบร้อย", use_column_width=True)
                st.title("STEP  3: นําไปนึ่ง ")
                st.write("เมื่อเตรียมไข่เสร็จเรียบร้อยแล้ว ก่อนนำนำไข่ไปนึ่งให้หยดน้ำเกลือนิดหน่อยเพื่อเพิ่มรสชาติให้กับไข่ นำไปนึ่งโดยใช้เวลาประมาณ 3-5 นาที ถ้านึ่งนานไปไข่จะแข็งทานไม่อร่อยครับ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/10.JPG', caption="หยอดน้ำเกลือ", use_column_width=True)
                with col2:
                    st.image('Recipe/06/11.JPG', caption="นําไปนึ่ง", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/06/12.JPG', caption="ใช้เวลานึ่ง 3-5 นาที", use_column_width=True)
                with col2:
                    st.image('Recipe/06/13.JPG', caption="สุกแล้ว", use_column_width=True) 
                st.image('Recipe/06/14.JPG', caption="พร้อมเสิร์ฟ", use_column_width=True)           
        ## ##
        if pred_id == 6:
            st.image('Nutrition/6.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/07/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : หุงข้าวให้เป็นสีฟ้า ")
                st.write("หุงข้าวให้สวยงาม เริ่มด้วยการล้างข้าวสารให้สะอาด เติมน้ำเปล่าลงไป จากนั้นใส่ดอกอัญชัน ขยี้ดอกอัญชันเล็กน้อยให้น้ำออกมาเป็นสีฟ้า แล้วหุงข้าวที่เราเตรียมไว้ไปหุงให้สุก")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/2.JPG', caption="ใส่ดอกอัญชัน", use_column_width=True)
                with col2:
                    st.image('Recipe/07/3.JPG', caption="หุงให้สุก", use_column_width=True)    
                
                st.title("STEP 2 : ทํานํ้าบูดูให้อร่อย ")
                st.write("ใส่น้ำบูดูลงในหม้อ ตามด้วยตะไคร้ ใบมะกรูด ข่า หอมแดง และปลาอินทรีเค็ม ตั้งไฟกลางเคี่ยวให้งวดลง ให้เหลือน้ำสัก ¾ ของน้ำบูดูเดิม")
                st.write("ปรุงรสด้วยน้ำตาลมะพร้าวแล้วเคี่ยวต่อสักพัก หมั่นคนนะคะ ระวังไม่ให้ไหม้ก้น เมื่อได้ที่แล้วยกลง กรองเอากากออก จะได้น้ำบูดูไว้ราดข้าวยำของเรา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/4.JPG', caption="ใส่นํ้าบูดูลงในหม้อ", use_column_width=True)
                with col2:
                    st.image('Recipe/07/5.JPG', caption="ปรุงรสด้วยนํ้าตาลมะพร้าว", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("เตรียมเครื่องทอปปิ้งที่เราต้องการเสิร์ฟ โดยซอยผักทั้งหมดเตรียมไว้")
                st.write("จัดจานให้สวยงาม เริ่มจากข้าวสวยอัญชันของเรา ตามด้วยน้ำบูดูที่เคี่ยวเสร็จแล้ว และทอปปิ้งผักทั้งหมด กุ้งแห้งโขลก มะพร้าวคั่ว และพริกแห้งป่น เท่านี้ก็พร้อมเสิร์ฟแล้วจ้า")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/6.JPG', caption="ผักสด ๆ น่าทานสุด ๆ", use_column_width=True)
                with col2:
                    st.image('Recipe/07/7.JPG', caption="ตักข้าวใส่จาน", use_column_width=True) 
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/8.JPG', caption="สวยงาม", use_column_width=True)
                with col2:
                    st.image('Recipe/07/9.JPG', caption="เวลาทานให้คลุกเคล้าเข้ากันเเบบนี้", use_column_width=True)
                st.image('Recipe/07/10.JPG', caption="พร้อมทานเเล้วครับ", use_column_width=True)
        
        ## ##
        if pred_id == 7:
            st.image('Nutrition/7.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/08/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ผัดพริกเเกง ")
                st.write("ตั้งกระทะโดยใช้ไฟกลาง พอกระทะร้อนใส่น้ำมันลงไปตามด้วยพริกแกงคั่วกลิ้ง ผัดให้มีกลิ่นหอม")
                st.image('Recipe/08/2.JPG', caption="ใส่พริกเเกงลงไปเเล้วผัดให้หอม", use_column_width=True)  
                
                st.title("STEP 2 : ใส่หมูสับ + ปรุงรส ")
                st.write("นำหมูสับที่เตรียมไว้ใส่ลงไปค่ะ ผัดให้หมูสับเข้ากับน้ำพริกแกง ระหว่างนี้เติมน้ำเปล่าต้มสุกที่เตรียมไว้ จะได้ผัดได้ง่ายขึ้น")
                st.write("เมื่อส่วนผสมเข้ากันดีแล้ว ปรุงรสด้วยผงปรุงรสและน้ำปลา ผัดให้เข้ากันอีกครั้ง")
                st.write("นำพริกไทยอ่อน ตะไคร้ซอย ใบมะกรูดซอย และพริกชี้ฟ้าแดง ใส่ลงไปผัดให้เข้ากัน เมื่อหมูสุกและทุกอย่างเข้ากันดี ปิดไฟได้เลย")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/08/3.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/08/4.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/08/5.JPG', caption="ปรุงรส", use_column_width=True)
                with col2:
                    st.image('Recipe/08/6.JPG', caption="เพิ่มความหอมด้วยใบมะกรูด", use_column_width=True)
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    
        if pred_id == 8:
            st.image('Nutrition/8.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 9:
            st.image('Nutrition/9.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    


        if pred_id == 10:
            st.image('Nutrition/10.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 11:
            st.image('Nutrition/11.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 12:
            st.image('Nutrition/12.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    


        if pred_id == 13:
            st.image('Nutrition/13.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 14:
            st.image('Nutrition/14.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 15:
            st.image('Nutrition/15.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    
        if pred_id == 16:
            st.image('Nutrition/16.PNG')
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    

        if pred_id == 17:
            st.image('Nutrition/17.PNG') 
            if st.button("เเนะนําวิธีการกิน"):
                st.image("example_picture.jpg", caption="Example Picture", use_column_width=True)
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/01/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ต้มสมุนไพร ")
                st.write("ตั้งน้ำให้เดือดจัด นำเครื่องสมุนไพรที่บุบเตรียมไว้ใส่ลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/2.JPG', caption="บุบสมุนไพรให้แหลก", use_column_width=True)
                with col2:
                    st.image('Recipe/01/3.JPG', caption="ต้มสมุนไพรให้มีกลิ่นหอม", use_column_width=True)    
                
                st.title("STEP 2 :  ใส่เนื้อไก่ ")
                st.write("นําเนื้อไก่ใส่ลงไป เเล้วเบาไฟ เพื่อให้นํ้าซุบใส ปรุงรสด้วยเกลือ นํ้าปลา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/4.JPG', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/01/5.JPG', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("คอยช้อนฟองออกเพื่อให้นํ้าใส เคี่ยวไปเรื่อยๆ จนเนื้อไก่เปื่อยนุ่ม เเละความหวานหอมจากสมุนไพรออกมาทั่วนํ้าซุป จึงค่อยๆตักเสิร์ฟร้อนๆ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/01/6.JPG', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)
                with col2:
                    st.image('Recipe/01/7.JPG', caption="ไก่นุ่มนํ้าเเกงหรอย ซดคล่องคอ!", use_column_width=True)    
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
