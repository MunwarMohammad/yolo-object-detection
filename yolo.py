import io
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ---------- Paths ----------
DEFAULT_CFG  = "yolov3-tiny.cfg"
DEFAULT_WEI  = "yolov3-tiny.weights"
DEFAULT_NAMES= "coco.names"

# ---------- Utils ----------
@st.cache_resource(show_spinner=False)
def load_classes(names_path: str):
    with open(names_path, "rt", encoding="utf-8") as f:
        return [c.strip() for c in f if c.strip()]

@st.cache_resource(show_spinner=True)
def load_net(cfg_path: str, weights_path: str):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # CPU target (change if you have CUDA build)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # get output layer names (works for all OpenCV versions)
    try:
        out_names = net.getUnconnectedOutLayersNames()
    except Exception:
        layer_names = net.getLayerNames()
        out_layers = net.getUnconnectedOutLayers()
        out_names = [layer_names[i[0]-1] for i in out_layers]
    return net, out_names

def make_blob(bgr_img, size=(416, 416)):
    blob = cv2.dnn.blobFromImage(
        bgr_img, 1/255.0, size, [0, 0, 0],
        swapRB=True, crop=False
    )
    return blob

def post_process(frame_bgr, outs, classes, conf_thresh=0.25, nms_thresh=0.45):
    H, W = frame_bgr.shape[:2]
    boxes, confidences, classIDs = [], [], []

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf >= conf_thresh:
                cx = int(det[0] * W)
                cy = int(det[1] * H)
                w  = int(det[2] * W)
                h  = int(det[3] * H)
                x  = int(cx - w/2)
                y  = int(cy - h/2)
                boxes.append([x, y, w, h])
                confidences.append(conf)
                classIDs.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    # Draw on a copy
    draw = frame_bgr.copy()
    if len(idxs) > 0:
        for i in np.array(idxs).flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[classIDs[i]]} {confidences[i]:.2f}"
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y_text = max(0, y-7)
            cv2.putText(draw, label, (x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return draw, idxs, boxes, confidences, classIDs

def pil_to_bgr(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb_img(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ---------- UI ----------
st.set_page_config(page_title="YOLOv3-tiny Streamlit", layout="centered")
st.title("YOLOv3-tiny Object Detection (OpenCV DNN)")

with st.sidebar:
    st.header("Model files")
    cfg_path   = st.text_input("Config (.cfg)", value=DEFAULT_CFG)
    weights_path = st.text_input("Weights (.weights)", value=DEFAULT_WEI)
    names_path = st.text_input("Class names (.names)", value=DEFAULT_NAMES)

    st.header("Inference settings")
    inp_size = st.selectbox("Input size", options=[(416,416), (320,320), (608,608)], index=0)
    conf_th  = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    nms_th   = st.slider("NMS IoU threshold",    0.0, 1.0, 0.45, 0.01)

# Validate paths
for p in [cfg_path, weights_path, names_path]:
    if not Path(p).exists():
        st.info("Put the model files next to app.py or provide full paths in the sidebar.")

# Load model & classes
try:
    classes = load_classes(names_path)
    net, out_names = load_net(cfg_path, weights_path)
except Exception as e:
    st.error(f"Failed to load model/classes: {e}")
    st.stop()

st.success(f"Loaded {len(classes)} classes · OpenCV {cv2.__version__}")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
run_btn = st.button("Run detection", type="primary")
# sample_col1, sample_col2 = st.columns(2)
# with sample_col1:
#     use_sample = st.checkbox("Use a sample dog+bike image (if you have one locally as ./images/dog.jpg)")
# with sample_col2:
    # run_btn = st.button("Run detection", type="primary")

# Pick image source
# img_bgr = None
# if uploaded is not None:
#     pil_img = Image.open(io.BytesIO(uploaded.read()))
#     img_bgr = pil_to_bgr(pil_img)
# elif use_sample and Path("./images/dog.jpg").exists():
#     img_bgr = cv2.imread("./images/dog.jpg")

img_bgr = None
if uploaded is not None:
    # Streamlit file object can go straight into PIL
    pil_img = Image.open(uploaded)
    img_bgr = pil_to_bgr(pil_img)
else:
    st.info("Please upload an image to run detection.")


if img_bgr is not None:
    st.image(bgr_to_rgb_img(img_bgr), caption="Input", use_container_width=True)
# else:
    # st.warning("Upload an image (or enable the sample) to run detection.")

if run_btn and img_bgr is not None:
    try:
        blob = make_blob(img_bgr, inp_size)
        net.setInput(blob)
        outs = net.forward(out_names)

        out_img_bgr, idxs, boxes, confs, classIDs = post_process(
            img_bgr, outs, classes, conf_thresh=conf_th, nms_thresh=nms_th
        )

        st.image(bgr_to_rgb_img(out_img_bgr), caption="Detections", use_container_width=True)

        # Show raw details
        with st.expander("Detections (after NMS)"):
            st.write(f"Kept boxes: {0 if len(idxs)==0 else len(np.array(idxs).flatten())}")
            rows = []
            if len(idxs) > 0:
                for i in np.array(idxs).flatten():
                    x, y, w, h = boxes[i]
                    rows.append({
                        "class_id": int(classIDs[i]),
                        "class": classes[classIDs[i]],
                        "confidence": float(confs[i]),
                        "box [x,y,w,h]": [int(x),int(y),int(w),int(h)]
                    })
            st.dataframe(rows, use_container_width=True)

    except Exception as e:
        st.error(f"Error during inference: {e}")
