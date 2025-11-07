import streamlit as st
import time

# Progress bar
st.header('st.progress')
st.caption('Display a progress bar')

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1)

st.success("Progress completed!")

# Spinner
st.header("st.spinner")
st.caption("Display a spinner while something is loading")

with st.spinner("Something is loading..."):
    time.sleep(3)
st.success("Loading complete!")

# Text Color
st.header("Text Color")
st.info("This is an information message")

st.header("Success")
st.success("This is success color")

st.header("Warning")
st.warning("This is warning color")

st.header("Error")
st.error("This is error color")

# Display image
st.header("Display image")
st.image(r"C:\Users\akilan\Downloads\download.png", caption="Nature", width=500)
st.image("download.png", caption="Nature", width=500)

# Display video
st.header("Display video")
video_location = open("./media/waterfalls.mp4", "rb")
video_bytes = video_location.read()
st.video(video_bytes)

# Display audio
st.header("Display audio")
audio_file = open("./media/sample_audio.mp3", "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/mp3")