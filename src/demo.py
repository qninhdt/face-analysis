import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent)

import cv2
import numpy as np
import streamlit as st

from detector import FaceDetector

detector = None

st.title("Face detector")


def update_nms_threshold():
    global detector
    if detector is not None:
        detector.nms_thre = nms_threshold


def update_conf_threshold():
    global detector
    if detector is not None:
        detector.conf_thre = conf_threshold


with st.sidebar:
    st.title("Settings")
    model = st.file_uploader("Choose a ONNX file", type="onnx")

    conf_threshold = st.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.05,
        on_change=update_conf_threshold,
        key="conf_threshold",
    )
    nms_threshold = st.slider(
        "NMS threshold",
        0.0,
        1.0,
        0.1,
        on_change=update_nms_threshold,
        key="nms_threshold",
    )

    if model is not None:
        with st.spinner("Loading model..."):
            detector = FaceDetector((model.read()), conf_threshold, nms_threshold)

    uploaded_file = st.file_uploader("Choose a image file", type="jpg")


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    with st.spinner("Wait for it..."):
        outputs, images = detector.detect_with_result_images(opencv_image)
        faces = outputs[0]
        image = images[0]

        st.image(image, channels="BGR")

    st.success("Done!")
