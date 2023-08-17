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
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/03/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เตรียมเครื่องแกงน้ำยากะทิ ")
                st.write("ต้มน้ำให้เดือด ใส่ตะไคร้ กระเทียม หอมแดง ข่า กระชาย และ พริกแห้งลงไป ตามด้วยเนื้อปลา ต้มจนสุกกรองทุกอย่างออก และเก็บน้ำต้มไว้ด้วย")
                st.write("นําผักเเละเนื้อปลาที่ต้ม มาโขลกให้ละเอียดจนกลายเป็นเครื่องเเกง")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/2.WEBP', caption="ใส่ตะไคร้ กระเทียม หอมเเดง ข่า กระชาย เเละพริกเเห้งลงไป เเละตามด้วยเนื้อปลา ต้มจนสุก", use_column_width=True)
                with col2:
                    st.image('Recipe/03/3.WEBP', caption="นําผัก เเละเนื้อปลาที่ต้มมาโขลกให้ละเอียด", use_column_width=True)    
                
                st.title("STEP 2 : ทํานํ้ายากะทิ ")
                st.write("ตั้งกระทะเทน้ำกะทิลงไปครึ่งหนึ่ง รอจนร้อนได้ที่ ให้นำเครื่องแกงลงไปผัดกับกะทิจนเข้ากันแล้ว ให้เติมกะทิส่วนที่เหลือลงไป ตามด้วยน้ำเปล่าเคี่ยวจนกะทิแตกมัน")
                st.write("ปรุงรสด้วยน้ำปลา เกลือ และตามด้วยใส่ลูกชิ้นปลาลงไป")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/4.WEBP', caption="นำเครื่องแกงลงไปผัดกับกะทิจนเข้ากัน", use_column_width=True)
                with col2:
                    st.image('Recipe/03/5.WEBP', caption="เติมกะทิส่วนที่เหลือลงไปในเครื่องแกงที่ผัดไว้ก่อนหน้านี้", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("ตักน้ำยากะทิราดบนขนมจีน เสิร์ฟคู่กับผักแกล้ม เพียงเท่านี้ “ขนมจีนน้ำยากะทิ” ของเราก็พร้อมรับประทานแล้ว")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/03/6.WEBP', caption="ตักน้ำยากะทิใส่จานเตรียมเสิร์ฟ", use_column_width=True)
                with col2:
                    st.image('Recipe/03/7.WEBP', caption="ขนมจีนนํ้ายากะทิ พร้อมรับประทานเเล้วครับ", use_column_width=True)    
        
        ## ไก่กอเเละ ##
        if pred_id == 3:
           st.image('Nutrition/3.PNG')
           if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/04/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ทําพริกเเกง ")
                st.write("เริ่มทำพริกแกง โดยการใส่พริกชี้ฟ้าแห้งแช่น้ำสับ พริกขี้หนูแห้งแช่น้ำสับ หอมแดง กระเทียม ลูกผักชี ยี่หร่า อบเชย และกะปิ ลงในครก แล้วโขลกให้ละเอียดเข้ากัน")
                st.write("ตั้งกระทะใส่น้ำมันพืชจนพอร้อน นำเครื่องแกงที่โขลกไว้ลงไปผัดจนหอม ใส่กะทิลงไปเคี่ยวจนแตกมัน")
                st.write("ปรุงรสด้วยน้ำมะขามเปียก น้ำตาลปี๊บ และน้ำปลา เคี่ยวต่อซักครู่แล้วยกลงพักไว้ให้เย็น")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/2.WEBP', caption="โขลกพริกแกงให้ละเอียด", use_column_width=True)
                with col2:
                    st.image('Recipe/04/3.JPG', caption="ใส่กะทิลงไปเคี่ยวจนแตกมัน", use_column_width=True)    
                
                st.title("STEP 2 : หมักไก่ ")
                st.write("โขลกกระเทียม ผงขมิ้น และเกลือป่น ให้ละเอียด")
                st.write("หมักสะโพกไก่หั่นชิ้น ด้วยส่วนผสมที่โขลกไว้ และกะทิ คลุกเคล้าให้เข้ากัน หมักไว้ 30 นาที")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/4.WEBP', caption="โขลกกระเทียม ผงขมิ้น และเกลือป่น ให้ละเอียด", use_column_width=True)
                with col2:
                    st.image('Recipe/04/5.JPG', caption="หมักไก่ไว้ 30 นาที", use_column_width=True)
                
                st.title("STEP  3: ย่าง ")
                st.write("นำไก่ที่หมักไว้มาเสียบไม้ แล้วนำขึ้นเตาย่าง คอยทาด้วยพริกแกงที่ทำไว้อย่างสม่ำเสมอทั้งสองด้าน ย่างจนสุกสวย จึงนำขึ้นจัดเสิร์ฟ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/04/6.WEBP', caption="นำไก่ที่หมักไว้มาเสียบไม้", use_column_width=True)
                with col2:
                    st.image('Recipe/04/7.WEBP', caption="ทาด้วยพริกแกงที่ทำไว้อย่างสม่ำเสมอทั้งสองด้าน", use_column_width=True)    
        
        ## ไก่ทอด ##
        if pred_id == 4:
            st.image('Nutrition/4.PNG')
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
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/06/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
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
                    st.image('Recipe/06/9.WEBP', caption="ใส่เรียบร้อย", use_column_width=True)
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
                st.image('Recipe/06/14.WEBP', caption="พร้อมเสิร์ฟ", use_column_width=True)           
        ## ##
        if pred_id == 6:
            st.image('Nutrition/6.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/07/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : หุงข้าวให้เป็นสีฟ้า ")
                st.write("หุงข้าวให้สวยงาม เริ่มด้วยการล้างข้าวสารให้สะอาด เติมน้ำเปล่าลงไป จากนั้นใส่ดอกอัญชัน ขยี้ดอกอัญชันเล็กน้อยให้น้ำออกมาเป็นสีฟ้า แล้วหุงข้าวที่เราเตรียมไว้ไปหุงให้สุก")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/2.WEBP', caption="ใส่ดอกอัญชัน", use_column_width=True)
                with col2:
                    st.image('Recipe/07/3.WEBP', caption="หุงให้สุก", use_column_width=True)    
                
                st.title("STEP 2 : ทํานํ้าบูดูให้อร่อย ")
                st.write("ใส่น้ำบูดูลงในหม้อ ตามด้วยตะไคร้ ใบมะกรูด ข่า หอมแดง และปลาอินทรีเค็ม ตั้งไฟกลางเคี่ยวให้งวดลง ให้เหลือน้ำสัก ¾ ของน้ำบูดูเดิม")
                st.write("ปรุงรสด้วยน้ำตาลมะพร้าวแล้วเคี่ยวต่อสักพัก หมั่นคนนะคะ ระวังไม่ให้ไหม้ก้น เมื่อได้ที่แล้วยกลง กรองเอากากออก จะได้น้ำบูดูไว้ราดข้าวยำของเรา")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/4.WEBP', caption="ใส่นํ้าบูดูลงในหม้อ", use_column_width=True)
                with col2:
                    st.image('Recipe/07/5.JPG', caption="ปรุงรสด้วยนํ้าตาลมะพร้าว", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("เตรียมเครื่องทอปปิ้งที่เราต้องการเสิร์ฟ โดยซอยผักทั้งหมดเตรียมไว้")
                st.write("จัดจานให้สวยงาม เริ่มจากข้าวสวยอัญชันของเรา ตามด้วยน้ำบูดูที่เคี่ยวเสร็จแล้ว และทอปปิ้งผักทั้งหมด กุ้งแห้งโขลก มะพร้าวคั่ว และพริกแห้งป่น เท่านี้ก็พร้อมเสิร์ฟแล้วจ้า")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/6.WEBP', caption="ผักสด ๆ น่าทานสุด ๆ", use_column_width=True)
                with col2:
                    st.image('Recipe/07/7.WEBP', caption="ตักข้าวใส่จาน", use_column_width=True) 
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/07/8.WEBP', caption="สวยงาม", use_column_width=True)
                with col2:
                    st.image('Recipe/07/9.WEBP', caption="เวลาทานให้คลุกเคล้าเข้ากันเเบบนี้", use_column_width=True)
                st.image('Recipe/07/10.WEBP', caption="พร้อมทานเเล้วครับ", use_column_width=True)
        
        ## คั่วกลิ้ง ##
        if pred_id == 7:
            st.image('Nutrition/7.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/08/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ผัดพริกเเกง ")
                st.write("ตั้งกระทะโดยใช้ไฟกลาง พอกระทะร้อนใส่น้ำมันลงไปตามด้วยพริกแกงคั่วกลิ้ง ผัดให้มีกลิ่นหอม")
                st.image('Recipe/08/2.WEBP', caption="ใส่พริกเเกงลงไปเเล้วผัดให้หอม", use_column_width=True)  
                
                st.title("STEP 2 : ใส่หมูสับ + ปรุงรส ")
                st.write("นำหมูสับที่เตรียมไว้ใส่ลงไปครับ ผัดให้หมูสับเข้ากับน้ำพริกแกง ระหว่างนี้เติมน้ำเปล่าต้มสุกที่เตรียมไว้ จะได้ผัดได้ง่ายขึ้น")
                st.write("เมื่อส่วนผสมเข้ากันดีแล้ว ปรุงรสด้วยผงปรุงรสและน้ำปลา ผัดให้เข้ากันอีกครั้ง")
                st.write("นำพริกไทยอ่อน ตะไคร้ซอย ใบมะกรูดซอย และพริกชี้ฟ้าแดง ใส่ลงไปผัดให้เข้ากัน เมื่อหมูสุกและทุกอย่างเข้ากันดี ปิดไฟได้เลย")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/08/3.WEBP', caption="ใส่เนื้อไก่ลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/08/4.WEBP', caption="ปรุงรสด้วยเกลือเเละนํ้าปลา", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/08/5.JPG', caption="ปรุงรส", use_column_width=True)
                with col2:
                    st.image('Recipe/08/6.JPG', caption="เพิ่มความหอมด้วยใบมะกรูด", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/08/7.WEBP', caption="เพิ่มสีสันด้วยพริกชี้ฟ้าเเดง", use_column_width=True)
                with col2:
                    st.image('Recipe/08/8.JPG', caption="ผัดให้เข้ากัน เเค่นี้ก็พร้อมทานเเล้ว", use_column_width=True)
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("นำคั่วกลิ้งหมูที่สุกแล้วจัดเสิร์ฟใส่จาน ตกแต่งด้วยใบโหระพา พริกชี้ฟ้าแดงซอย และใบมะกรูดซอย กินกับผักเครื่องเคียงตามชอบได้เลยครับ")
                st.image('Recipe/08/9.WEBP', caption="คอยช้อนฟองออกเพื่อให้นํ้าใส", use_column_width=True)

        ## ใบเหลียงต้มกะทิ ##        
        if pred_id == 8:
            st.image('Nutrition/8.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/09/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เตรียมเครื่องเเกง ")
                st.write("นำใบเหลียงที่เตรียมไว้ล้างน้ำให้สะอาด สะบัดให้สะเด็ดน้ำ เลือกเอาใบอ่อน ๆ ไม่แก่ จะใช้มีดหรือกรรไรตัดให้เล็กลงก็ได้")
                st.write("นำกุ้งสดล้างน้ำให้สะอาด จากนั้นปอกเปลือกออก แล้วผ่าหลังดึงเส้นดำออก ล้างน้ำอีกครั้งพักไว้")
            
                
                st.title("STEP 2 : โขลกส่วนผสม ")
                st.write("นำหอมแดง กะปิ และพริกไทยขาวมาโขลกให้ละเอียด")
                st.image('Recipe/09/2.WEBP', caption="ทำการโขลก หอมแดง กะปิ และพริกไทยขาว", use_column_width=True)
                

                st.title("STEP  3: ลงมือเเกง ")
                st.write("ตั้งหม้อโดยใช้ไฟกลาง ใส่หัวกะทิลงไป รอให้กะทิเดือดอ่อน ๆ ใส่เครื่องแกงที่โขลกไว้ลงไปคนให้เข้ากันค่ะ จากนั้นใส่กุ้งแห้งตามลงไป รอจนกะทิเดือดอีกครั้ง")
                st.write("หลังจากนั้นเติมหางกะทิลงไปอีก รอจนน้ำแกงเดือดอีกครั้ง")
                st.write("เมื่อกะทิเดือดแล้วนำกุ้งสดใส่ลงไป รอจนกุ้งสุกดีแล้ว จึงใส่ใบเหลียงลงไป")
                st.write("ใช้ทัพพีกดให้ใบเหลียงจมน้ำแกงให้หมด รอให้ใบเหลียงอ่อนตัวลง ทำการปรุงรสด้วยน้ำตาลทรายและเกลือป่น รอให้น้ำแกงเดือด โดยเราจะใช้ไฟกลางนะครับ")
                st.write("รอให้นำแกงเดือดอ่อน ๆ อีกครั้ง แล้วปิดไฟได้เลย")             
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/09/3.WEBP', caption="ใส่เครื่องเเกงที่โขลกไว้", use_column_width=True)
                with col2:
                    st.image('Recipe/09/4.WEBP', caption="ตามด้วยหางกะทิ", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/09/5.WEBP', caption="ใส่ใบเหลียง", use_column_width=True)
                with col2:
                    st.image('Recipe/09/6.WEBP', caption="เรียบร้อยครับ", use_column_width=True)    

                st.title("STEP 4: จัดเสิร์ฟ ")
                st.write("นำใบเหลียงต้มกะทิตักใส่ชาม พร้อมเสิร์ฟได้เลย")
                st.image('Recipe/09/7.WEBP', caption="เตรียมจัดเสิร์ฟได้เลย", use_column_width=True)


        ####
        if pred_id == 9:
            st.image('Nutrition/9.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/10/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : คั่วอบเชย โป๊ยกั๊ก พริกไทยดำ-ขาว ")
                st.write("โดยการให้มีกลิ่นหอม แล้วนำมาตำให้ละเอียด ใส่รากผักชี กระเทียม และเกลือนิดหน่อย ลงไปตำให้เข้ากันสำหรับหมักหมูสามชั้น ")
                st.image('Recipe/10/2.JPG', caption="คั่วอบเชย โป๊ยกั๊ก พริกไทยดำ-ขาว ", use_column_width=True)
               
                
                st.title("STEP 2 : หมูสามชั้นหั่นชิ้นหนาประมาณ 2นิ้ว ")
                st.write("หมักให้เข้าเนื้อด้วยเครื่องที่ตำไว้ ปรุงรสด้วยซีอิ้วขาว ซีอิ้วดำ น้ำตาลทรายแดง น้ำตาลปิ๊ป และเกลือเล็กน้อย ")
                st.image('Recipe/10/3.JPG', caption="หมักหมูให้เข้ากัน", use_column_width=True)
                
                st.title("STEP 3: ผัดหมูสามชั้นที่หมักไว้ ")
                st.write("ผัดหมูสามชั้นที่หมักไว้ในกระทะให้พอผิวด้านนอกสุกนิดหน่อย.. แล้วเติมน้ำลงไปใส่ซุปหมูก้อนคนอร์เพิ่มความหอมกลมกล่อม..เคี่ยวให้หมูเปื่อยนุ่มดีใช้เวลาประมาณ 1ชม. น้ำจะงวดลงเข้าเนื้อหมูเข้มข้น..หอมหวลชวนกินเป็นอันเสร็จ ")
                
                st.image('Recipe/10/4.JPG', caption="ผัดให้เข้ากัน", use_column_width=True)    


                st.title("STEP 4 : จัดเสิร์ฟ ")
                st.write("หมูฮ้อง หอม ๆ รสชาติเค็มนำหวาน หอมกลิ่นเครื่องเทศต่าง ๆ เปื่อยนุ่มกำลังดี ")
                st.image('Recipe/10/5.JPG', caption="หมูฮ้องพร้อมเสิร์ฟเเล้ว", use_column_width=True)

        ####
        if pred_id == 10:
            st.image('Nutrition/10.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/11/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : โขลกเครื่องให้ละเอียด ")
                st.write("ใส่พริกขี้หนู พริกแดง กระเทียม น้ำตาลทราย ลงไปตำให้แหลก ตามด้วยกะปิ ตำให้เข้ากัน")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/11/2.WEBP', caption="ใส่พริกขี้หนู พริกแดง กระเทียม น้ำตาลทราย ลงไปตำ", use_column_width=True)
                with col2:
                    st.image('Recipe/11/3.WEBP', caption="ใส่กะปิตามลงไป ตำให้เข้ากัน", use_column_width=True)    
                
                st.title("STEP 2 : ปรุงรสน้ำพริก ")
                st.write("ปรุงรสน้ำพริกกะปิด้วยน้ำปลา และน้ำมะนาว")
                st.write("ใส่มะเขือพวงลงไป เท่านี้ก็เสร็จ จัดเสิร์ฟได้เลย")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/11/4.WEBP', caption="ปรุงรสด้วยน้ำปลาและน้ำมะนาว", use_column_width=True)
                with col2:
                    st.image('Recipe/11/5.WEBP', caption="จัดเสิร์ฟได้เลย", use_column_width=True)
                
        ####
        if pred_id == 11:
            st.image('Nutrition/11.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/12/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ตำ ")
                st.write("ห่อกะปิด้วยอะลูมิเนียมฟอยล์ แล้วนำไปนาบกับกระทะด้วยไฟอ่อน จนส่งกลิ่นหอม")
                st.write("ใส่หอมแดง กระเทียม และพริกขี้หนู ลงในครก ตำให้ละเอียดเข้ากัน")
                st.write("นำกะปิที่นาบกระทะไว้ใส่ลงไปในครก ตามด้วยน้ำปลา น้ำตาลปี๊บ และน้ำมะนาว คลุกให้เข้ากัน")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/12/2.JPG', caption="ใส่หอมแดง กระเทียม และพริกขี้หนู ลงในครก", use_column_width=True)
                with col2:
                    st.image('Recipe/12/3.JPG', caption="ตำให้ละเอียดเข้ากัน", use_column_width=True)    
                
                st.title("STEP 2 : คลุก ")
                st.write("นำกุ้งเสียบลงไปคั่วในกระทะไฟกลาง จนกุ้งเสียบเริ่มกรอบ แล้วนำขึ้น")
                st.write("ตักกุ้งเสียบที่คั่วไว้ลงในครก คลุกให้เข้ากัน")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/12/4.JPG', caption="คั่วกุ้งเสียบด้วยไฟกลาง", use_column_width=True)
                with col2:
                    st.image('Recipe/12/5.JPG', caption="คลุกให้เข้ากัน", use_column_width=True)
                
                st.title("STEP  3: จัดเสิร์ฟ ")
                st.write("ตักน้ำพริกที่คลุกเรียบร้อยแล้วขึ้นจัดเสิร์ฟพร้อมกับผักสดตามชอบ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/12/6.JPG', caption="ตักขึ้นใส่ชาม", use_column_width=True)
                with col2:
                    st.image('Recipe/12/7.WEBP', caption="เสิร์ฟกับผักสดตามชอบ", use_column_width=True)    

        if pred_id == 12:
            st.image('Nutrition/12.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/13/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
                st.write("1. ตำกระเทียมกับขมิ้นให้พอแหลกครับ (ขมิ้นจะมียางหน่อยๆ เวลาตำแล้วจะรู้สึกว่าตำยากนิดๆ ครับ แล้วครกจะเหลืองติดสีไปซักพักเลยครับ)")
                st.write("2.เอาเครื่องทำตำแล้วกับเกลือเคล้ากับตัวปลาให้ทั่ว ทั้งข้างนอก ข้างใน แล้วพักไว้ 15 นาที")
                st.write("3. นำลงทอดในกระทะใส่น้ำมันมากหน่อยตั้งไฟให้ร้อนจัด")
                st.write("4. ใช้มือรูดเครื่องที่หมักออกจากตัวปลาบ้าง (ถ้าใส่ลงไปพร้อมกับเครื่องหมักจะไหม้ก่อนปลาสุกครับ) นำลงทอดจะสุก เหลือง ถ้าชอบกินเครื่องหมักทอดให้นำเครื่องหมักลงลงทอดจนเหลืองกรอบอีกครั้ง")
           

        if pred_id == 13:
            st.image('Nutrition/13.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/14/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เตรียมวัตถุดิบ ")
                st.write("ล้างปลากะพงให้สะอาด ก่อนนำมาหั่นเป็นท่อน ๆ สำหรับนำไปต้ม")
                st.write("ปอกเปลือกสับปะรดแล้วหั่นเป็นแว่น ๆ พักเอาไว้")
                st.write("ปอกเปลือกมะละกอออก จากนั้นหั่นมะละกอเป็นชิ้นพอคำ")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/14/2.JPG', caption="ปอกเปลือกสับปะรด", use_column_width=True)
                with col2:
                    st.image('Recipe/14/3.JPG', caption="ปอกตาออก", use_column_width=True)    
                
                st.title("STEP 2 : ต้ม ")
                st.write("ตั้งน้ำสต๊อกไก่ใส่ลงในหม้อ นำขึ้นตั้งไฟโดยใช้ไฟกลาง รอจนน้ำเริ่มเดือด เมื่อน้ำเดือดให้ใส่พริกแกงเหลือง คนจนพริกแกงละลาย จากนั้นใส่ปลากะพงลงไป")
                st.write("เมื่อเดือดแล้ว ใส่มะละกอ และสับปะรดลงไป ต้มต่อจนสุก")
                st.write("ปรุงรสด้วยน้ำมะขามเปียก น้ำปลา เกลือ และน้ำตาล คนให้เข้ากันตั้งไฟต่อจนเดือดอีกรอบ")
                st.write("ปิดไฟแล้วเติมน้ำมะนาวลงไป เท่านี้ก็จัดเสิร์ฟได้เลยจ้า")
                st.write("TIP... ใส่น้ำมะนาวหลังจากปิดไฟแล้ว เพื่อให้รสชาติของน้ำมะนาวไม่ขม")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/14/4.JPG', caption="ใส่พริกแกงเหลือง", use_column_width=True)
                with col2:
                    st.image('Recipe/14/5.JPG', caption="ใส่เนื้อปลากะพง", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/14/4.JPG', caption="ใส่สับปะรด", use_column_width=True)
                with col2:
                    st.image('Recipe/14/5.JPG', caption="ปรุงรสด้วยน้ำปลา", use_column_width=True)
                
                st.image('Recipe/14/6.JPG', caption="พร้อมเสิร์ฟเเล้วครับ", use_column_width=True)      

        ####
        if pred_id == 14:
            st.image('Nutrition/14.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/15/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : เตรียมใบเหลียงให้เรียบร้อย ")
                st.write("เด็ดใบเหลียงเลือกใบที่อ่อนๆ ไม่แข็งกระด้าง ใบที่ใหญ่ให้ฉีกออกแต่ไม่ต้องเล็กมาก ล้างน้ำให้สะอาดพักสะเด็ดน้ำ")
                st.title("STEP 2 : นําลงไปผัด")
                st.write("นำน้ำมันใส่กะทะตั้งไฟให้ร้อน เจียวกระเทียมให้หอม ใส่ไข่ไก่ที่ตีไว้ผัดพอไข่เป็นวุ้น")
                st.title("STEP 3 : จัดเสิร์ฟ")
                st.write("ใส่ใบเหลียงตามด้วยเครื่องปรุงผัดให้ใบเหลียงพอสลบ ตัดใส่จานทานได้ทันที ใครจะกินกับน้ำพริกกะปิก็เข้ากัน")
                st.image('Recipe/15/2.JPG', caption="พร้อมเสิร์ฟเเล้วครับ", use_column_width=True)
                
            

        if pred_id == 15:
            st.image('Nutrition/15.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/16/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : โขลกพริกกระเทียม ")
                st.write("นำพริก กระเทียม หอมแดงซอย และกะปิ โขลกให้เป็นเนื้อเดียวกัน พักไว้")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/16/2.JPG', caption="ใส่กระเทียมลงไปโขลก", use_column_width=True)
                with col2:
                    st.image('Recipe/16/3.JPG', caption="โขลกพริก กระเทียม หอมแดงซอย และกะปิจนเข้ากัน", use_column_width=True)    
                
                st.title("STEP 2 : ตั้งกะทะ ผัดส่วนผสม ")
                st.write("นำส่วนผสมที่โขลกไว้แล้ว ลงไปผัดในกระทะ ใช้ไฟกลาง ผัดจนมีกลิ่นหอมเลย")
                st.write("หลังจากนั้นใส่เนื้อหมูลงไป ผัดจนหมูเริ่มสุก ระหว่างนี้เติมน้ำเปล่าต้มสุก เพื่อให้มีน้ำขลุกขลิกค่ะ ผัดให้ส่วนผสมทุกอย่างเข้ากันดี")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/16/4.JPG', caption="นำส่วนผสมที่โขลกไว้ลงไปผัด", use_column_width=True)
                with col2:
                    st.image('Recipe/16/5.JPG', caption="เติมน้ำเปล่าให้มีน้ำขลุกขลิก", use_column_width=True)
                
                st.title("STEP 3: ปรุงรส ")
                st.write("ปรุงรสด้วยน้ำตาลปี๊บ บีบน้ำมะนาวลงไป")
                st.write("ตามด้วยตะไคร้ซอย พริกชี้ฟ้า ใบมะกรูดซอย ผัดให้เข้ากันอีกครั้ง ปิดไฟได้เลย")
                st.write("นำหมูผัดกะปิตักใส่จาน ตกแต่งด้วยพริกชี้ฟ้า จัดขึ้นโต๊ะ พร้อมเสิร์ฟได้เลย")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/16/6.JPG', caption="ใส่ใบมะกรูดตามลงไป", use_column_width=True)
                with col2:
                    st.image('Recipe/16/7.JPG', caption="จัดเสิร์ฟได้เลย", use_column_width=True)    
        
        if pred_id == 16:
            st.image('Nutrition/16.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/17/1.WEBP', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            st.title("วิธีทํา")
            st.write("ทำพริกแกงสำหรับผัดสะตอ โดยการใส่กระเทียมลงไปในครก ตามด้วยพริกขี้หนู กะปิ และน้ำตาลปี๊บ โขลกให้ละเอียด ตักใส่ถ้วยพักไว้")
            st.write("นำกระทะตั้งไฟ ความร้อนปานกลาง เทน้ำมันพืชลงไป รอน้ำมันร้อน ใส่พริกแกงที่เตรียมไว้ ผัดจนหอม จากนั้นใส่กุ้งลงไปผัดจนสุก")
            st.write("ใส่สะตอแกะผ่าครึ่ง ผัดจนสุก จากนั้นปรุงรสด้วยน้ำปลา และมะนาวค่ะ ผัดต่อให้เข้ากันดี")
            st.image('Recipe/17/2.WEBP', caption="พร้อมเสิร์ฟเเล้วครับ", use_column_width=True)
           

        if pred_id == 17:
            st.image('Nutrition/17.PNG')
            if st.button("วิธีการปรุงอาหาร"):
                st.image('Recipe/18/1.JPG', caption="วัตถุดิบที่ใช้ในการทําอาหาร", use_column_width=True)
            
                st.title("STEP 1 : ตำพริกแกงไตปลา ")
                st.write("ใส่พริกแห้งตำให้ละเอียด แล้วใส่ผิวมะกรูด ตะไคร้สับ และขมิ้น โขลกให้ละเอียด ตามด้วยหอมแดง กระเทียม ส่วนพริกไทยขาว พริกไทยดำ ให้โขลกแยกแล้วใส่ทีหลังสุด ตำเครื่องแกงทั้งหมดให้ละเอียด เข้ากันดี")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/18/2.WEBP', caption="โขลกพริกแห้งให้ละเอียด", use_column_width=True)
                with col2:
                    st.image('Recipe/18/3.WEBP', caption="ใส่พริกไทยดำที่โขลกแยกไว้", use_column_width=True)    
                
                st.title("STEP 2 : ต้มไตปลา ")
                st.write("ตั้งหม้อ ใส่หอมแดง ข่า ตะไคร้ ใบมะกรูดฉีก มะขามเปียก ตามด้วยไตปลา และน้ำเล็กน้อย พอเดือดได้ที่ยกขึ้น กรองแล้วพักไว้")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/18/4.WEBP', caption="ต้มไตปลากับสมุนไพรเพิ่อดับกลิ่นคาว", use_column_width=True)
                with col2:
                    st.image('Recipe/18/5.WEBP', caption="กรอง", use_column_width=True)
                
                st.title("STEP  3: ต้มเเกงไตปลา ")
                st.write("ตั้งหม้อต้มน้ำให้เดือด ตามด้วยเครื่องแกงที่ตำไว้ และกะปิคนให้เข้ากัน ตามด้วย ใบมะกรูด ไตปลาที่ต้มไว้ พอเดือดใส่หน่อไม้ มะเขือเปราะ เนื้อปลาย่าง ถั่วฝักยาว พอเดือดอีกครั้งให้ปิดไฟ ตักใส่ชามจัดเสิร์ฟ เป็นอันเสร็จเรียบร้อย")
                col1, col2 = st.columns(2)
                with col1:
                    st.image('Recipe/18/6.WEBP', caption="ใส่มะเขือลงไปต้ม", use_column_width=True)
                with col2:
                    st.image('Recipe/18/7.WEBP', caption="ตักใส่ชามเสิร์ฟ", use_column_width=True)    
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
