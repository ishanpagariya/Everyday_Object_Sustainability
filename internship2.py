import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google import genai

client = genai.Client(api_key=["GEMINI_API_KEY"])

model = keras.models.load_model("keras_model.h5", compile=False)


st.title("Everyday Object Sustainability ♻️")

image = st.camera_input("Take a photo of waste")

if image:

    img = Image.open(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing waste..."):
        prediction = model.predict(img_array)

    labels = ["Highly sustainable","Moderately sustainable","Non sustainable","Compostable"]

    class_index = np.argmax(prediction)
    predicted_class = labels[class_index]

    st.success(f"This looks like it is: **{predicted_class}**")

    with st.spinner("Thinking hmmmmmmmmm..."):
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=(
                f"Explain in one sentence how {predicted_class} can be treated"
                
            )
        )

    if response and hasattr(response, "candidates") and response.candidates:
     text = response.candidates[0].content.parts[0].text
     st.write(text)
    else:
        st.write("Could not generate a response. Try again.")

              