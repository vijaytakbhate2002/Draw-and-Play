import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import joblib
import time
import base64
from PIL import Image 

model = joblib.load("CNN Digits Classification")
# HTML
st.markdown("""
<style>
.heading{
  font-family: cursive;
  font-size: 30px;
  color: rgb(245, 175, 71);
  font-weight:bold;
  margin: center;
}
.subheading{
  font-size: 20px;
  font-family: cursive;
  color: rgb(245, 175, 71);

}
.result{
  font-family: cursive;
  font-size: 20px;
  color: rgb(226, 245, 149); 
  font-weight:bold;
  margin: center; 
}
</style>""",unsafe_allow_html=True)

songs = {
  0:"Dil Galti Kar Baitha Hai(PagalWorld.com.se).mp3",
  1:"Hara Hara Shambhu(PagalWorld.com.se).mp3",
  2:"Hum Nashe Mein Toh Nahin(PagalWorld.com.se).mp3",
  3:"Kesariya(PagalWorld.com.se).mp3",
  4:"Maine Tera Naam Dil Rakh Diya(PagalWorld.com.se).mp3",
  5:"Mehbooba Main Teri Mehbooba (KGF Chapter 2)(PagalWorld.com.se).mp3",
  6:"Oh Humnasheen Yasser Desai 320 Kbps.mp3",
  7:"Saami Saami(PagalWorld.com.se).mp3",
  8:"Shiv Tandav Stotram.mp3",
  9:"Teri Mitti (mp3download.minewap.com).mp3"
}

# sidebar
st.sidebar.write('''<p class=result> Songs List </p>''',unsafe_allow_html=True)
df = pd.DataFrame(["Dil Galti","Hara Hara Shambhu","Hum Nashe Mein","Kesariya","Maine Tera Naam","Main Teri Mehbooba","Oh Humnasheen","Saami Saami","Shiv Tandav","Teri Mitti"],columns=["Songs"])
st.sidebar.write(df)

# main area
st.write("""<p class=heading> Draw and Play Song</p>""",unsafe_allow_html=True)
with st.expander("Adjust stroke Width"):
  stroke_width = st.slider('',1,30,19)
st.markdown("""<p class=subheading> Draw a digit below </p>""",unsafe_allow_html=True)

canvas_result = st_canvas(
    stroke_width = stroke_width,
    stroke_color = "#FFFFFF",
    background_color = "#000000",
    height = 400,
    width = 306,
    key="full_app",
)

def get_corners(img):
    flag1,height1 = False,0
    flag2,height2 = False,0
    flag3,width1 = False,0
    flag4,width2 = False,0

    for i,row in enumerate(img):
        if row.max() == 0 and flag1 != True:
            height1 += 1 
        else:
            flag1 = True
        if img[img.shape[0]-1-i].max() == 0 and flag2 != True:
            height2 += 1
        else:
            flag2 = True

    for i in range(img.shape[1]):
        if img[:,i:i+1].max() == 0 and flag3 != True:
            width1 += 1
        else:
            flag3 = True
        if img[:,img.shape[1]-1-i:img.shape[1]-i].max() == 0 and flag4 != True:
            width2 += 1 
        else:
            flag4 = True
    return height1,img.shape[0]-height2,width1,img.shape[1]-width2

def autoplay_song(result):
  with open(songs[result],"rb") as file:
      audio = file.read()
      mymidia_placeholder = st.empty()
      mymidia_str = "data:audio/ogg;base64,%s"%(base64.b64encode(audio).decode())
      mymidia_html = """
                        <audio autoplay class="stAudio">
                        <source src="%s" type="audio/ogg">
                        Your browser does not support the audio element.
                        </audio>
                      """%mymidia_str
      mymidia_placeholder.empty()
      time.sleep(0.1)
      mymidia_placeholder.markdown(mymidia_html, unsafe_allow_html=True)

def image_croper(image,padd):
  corners = get_corners(image)
  if min(corners) > padd:
    return image[corners[0]-padd:corners[1]+padd,corners[2]-padd:corners[3]+padd]
  else:
    return image[corners[0]:corners[1],corners[2]:corners[3]]
  

if canvas_result.image_data is not None: 
  image1 = canvas_result.image_data.astype('uint8')
  image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

  if image1.max() > 200:
    image1 = cv2.resize(image1,(28,28))
    image1 = image_croper(image1,2)
    image1 = cv2.resize(image1,(28,28))
    st.sidebar.image(image1,width=100)
    image1 = np.reshape(image1,(1,28,28,1))/255
    result = np.argmax(model.predict(image1))
    st.write(f"Song Number {result} is playing...")
    st.sidebar.write(f"""<p class=result> Result = {result} </p>""",unsafe_allow_html=True)

    null = autoplay_song(result)

with st.expander("Here is Demo"):
  demo = Image.open("demo.png")
  st.image(demo)
  

