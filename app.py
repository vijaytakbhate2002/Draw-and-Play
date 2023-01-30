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
  font-size: 30px;
  color: rgb(226, 245, 149); 
  font-weight:bold;
  margin: center; 
}
</style>""",unsafe_allow_html=True)

songs = {
  0:"Musics//Dil Galti Kar Baitha Hai(PagalWorld.com.se).mp3",
  1:"Musics//Hara Hara Shambhu(PagalWorld.com.se).mp3",
  2:"Musics//Hum Nashe Mein Toh Nahin(PagalWorld.com.se).mp3",
  3:"Musics//Kesariya(PagalWorld.com.se).mp3",
  4:"Musics//Maine Tera Naam Dil Rakh Diya(PagalWorld.com.se).mp3",
  5:"Musics//Mehbooba Main Teri Mehbooba (KGF Chapter 2)(PagalWorld.com.se).mp3",
  6:"Musics//Oh Humnasheen Yasser Desai 320 Kbps.mp3",
  7:"Musics//Saami Saami(PagalWorld.com.se).mp3",
  8:"Musics//Shiv Tandav Stotram.mp3",
  9:"Musics//Teri Mitti (mp3download.minewap.com).mp3"
}

# sidebar
st.sidebar.write('''<p class=result> Songs List </p>''',unsafe_allow_html=True)
df = pd.DataFrame(["Dil Galti","Hara Hara Shambhu","Hum Nashe Mein","Kesariya","Maine Tera Naam","Main Teri Mehbooba","Oh Humnasheen","Saami Saami","Shiv Tandav","Teri Mitti"],columns=["Songs"])
st.sidebar.write(df)

st.write("""<p class=heading> Draw and Play Song</p>""",unsafe_allow_html=True)
st.markdown("""<p class=subheading> Draw a digit below </p>""",unsafe_allow_html=True)

canvas_result = st_canvas(
    stroke_width = 19,
    stroke_color = "#FFFFFF",
    background_color = "#000000",
    height = 400,
    width = 306,
    key="full_app",
)

# ValueError: Layer 'conv2d_10' expected 2 variables, but received 0 variables during loading. Names of variables received: []

if canvas_result.image_data is not None: 
  image1 = canvas_result.image_data.astype('uint8')
  image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
  if image1.max() > 200:
    image1 = cv2.resize(image1,(28,28))
    st.sidebar.image(image1,width=100)
    image1 = np.reshape(image1,(1,28,28,1))/255
    result = np.argmax(model.predict(image1))
    st.write(f"Song Number {result} is playing...")
    st.sidebar.write(f"""<p class=result> Result = {result} </p>""",unsafe_allow_html=True)
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
with st.expander("Here is Demo"):
  demo = Image.open("demo.png")
  st.write("Please try to Draw Digit in full Drawbox")
  st.image(demo)
  

