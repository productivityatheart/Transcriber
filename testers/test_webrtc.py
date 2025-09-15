# test_webrtc.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.title("Simple Mic Test")
webrtc_streamer(
    key="test",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"audio": True, "video": False},
)
