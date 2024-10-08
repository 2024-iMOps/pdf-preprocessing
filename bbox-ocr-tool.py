import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ko'], gpu=True)

# Function to recognize text within bounding box
def recognize_text(image, coords):
    if coords:
        x1, y1, x2, y2 = coords
        cropped_img = image.crop((x1, y1, x2, y2))
        # Convert PIL image to OpenCV format for OCR
        cropped_img_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        result = reader.readtext(cropped_img_cv)
        extracted_text = "\n".join([res[1] for res in result])
        return extracted_text
    return "No bounding box selected."

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("Bounding Box Tool with OCR")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# CSV file to store bounding box data
csv_file = "/workspace/pdf-preprocessing/bounding_boxes.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["x1", "y1", "x2", "y2", "recognized_text", "label"]).to_csv(csv_file, index=False)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Resize the image to fit the container width while maintaining aspect ratio
    container_width = 800  # Set a fixed width for container
    display_width = min(container_width, image.width)
    display_height = int((display_width / image.width) * image.height)

    # Layout for image and bounding box information side by side
    col1, col2 = st.columns(2)

    with col1:
        # Create a canvas for drawing bounding boxes
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=image.resize((display_width, display_height)),
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas",
        )

    with col2:
        # Store the current bounding box data globally to show once per draw
        if "current_bbox" not in st.session_state:
            st.session_state.current_bbox = None
        if "bbox_data" not in st.session_state:
            st.session_state.bbox_data = []

        # If a bounding box is drawn, process it
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            scale_x = image.width / display_width
            scale_y = image.height / display_height
            obj = canvas_result.json_data["objects"][-1]  # Get the last drawn bounding box
            if obj["type"] == "rect":
                x1 = int(obj["left"] * scale_x)
                y1 = int(obj["top"] * scale_y)
                x2 = x1 + int(obj["width"] * scale_x)
                y2 = y1 + int(obj["height"] * scale_y)
                coords = [x1, y1, x2, y2]

                # Recognize text within the bounding box
                extracted_text = recognize_text(image, coords)
                st.session_state.current_bbox = {
                    "coords": coords,
                    "extracted_text": extracted_text
                }

        # Display the current bounding box details
        if st.session_state.current_bbox:
            coords = st.session_state.current_bbox["coords"]
            extracted_text = st.session_state.current_bbox["extracted_text"]

            # Input label for the bounding box
            label_options = ["text", "table", "image", "head1", "head2", "head3"]
            label = st.radio("Label for bounding box", label_options, key="label_radio")
            custom_label = st.text_input("Or enter a custom label", key="label_custom")
            final_label = custom_label if custom_label else ""

            # Display recognized text and coordinates
            st.write(f"Bounding Box Coordinates: {coords}")
            st.text_area("Recognized Text", extracted_text, key="text_area")

            # Add a save button
            if st.button("Save Bounding Box", key="save_button"):
                if final_label or label:
                    final_label = final_label if final_label else label
                    # Append data to list for saving
                    st.session_state.bbox_data.append({
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3],
                        "recognized_text": extracted_text,
                        "label": final_label
                    })
                    # Save bounding box data to CSV
                    df = pd.DataFrame(st.session_state.bbox_data)
                    df.to_csv(csv_file, mode='a', header=False, index=False)
                    st.success(f"Bounding box data saved to {csv_file}")
                    # Clear the current bounding box after saving
                    st.session_state.current_bbox = None