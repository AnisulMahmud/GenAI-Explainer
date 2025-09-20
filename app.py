import streamlit as st
from text_gen import make_script
from speech_gen import synthesize_long   
import soundfile as sf
import numpy as np

st.set_page_config(page_title="GeneraVoice – Paper to Voice", page_icon="🎙")

st.title("🎙 GeneraVoice")
st.caption("Paste a research abstract ")

#  User input 
abstract = st.text_area("Research abstract", height=200, placeholder="Paste your abstract here...")

if st.button("Generate Summary Script"):
    if not abstract.strip():
        st.warning("Please paste an abstract first.")
    else:
        with st.spinner("Generating narration script..."):
            script = make_script(abstract)
        st.session_state["script"] = script
        st.success("Script generated!")

# Showing script.summary if present
if "script" in st.session_state:
    st.subheader("📝 Generated Script")
    st.write(st.session_state["script"])

    if st.button("Generate Audio"):
        with st.spinner("Synthesizing voice..."):
            sr, audio = synthesize_long(st.session_state["script"]) 
            sf.write("output.wav", audio, sr)  # save to file
        st.success("Audio generated!")
        st.audio("output.wav")
        with open("output.wav", "rb") as f:
            st.download_button("Download Audio", f, file_name="explainer.wav", mime="audio/wav")
