# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 02:31:37 2024

@author: caglar.gurkan
"""
# Standard library imports
import os
import sys
import time
import traceback
from os.path import join
from collections import deque
import json
import threading

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the CollapsibleBox
from collapsible_box import CollapsibleBox

# Third-party imports
import numpy as np
import joblib
import torch
import cv2
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
import gradio as gr
import webbrowser
import socket             
import celldetection as cd
from ultralytics import YOLO
from cell.util import imread, imsave, copy_skimage_data
from scipy import signal  # For signal processing
from scipy.signal import butter, lfilter, iirnotch
# PyQt6 imports
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import QMessageBox, QFileDialog
# Local imports
from cell.cpn import CpnInterface
from cell.prep import multi_norm
from celldetection import label_cmap, to_h5, data, __version__
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
pg.setConfigOption('background', '#1e1e2e')
# Load a pretrained YOLOv8n model
kidney_CV_model = YOLO(r"./weights/s_model/best.pt")
kidney_ML_model = joblib.load(r"./ML_works/kidney_ml_model.pkl")



def torch_compile(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

torch.compile = torch_compile  # temporary workaround
default_model = 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
default_score_thresh = .9
default_nms_thresh = np.round(np.pi / 10, 4)
default_samples = 128
default_order = 5

def predict(
        img, model=default_model,
        enable_score_threshold=True, score_threshold=.1,
        enable_nms_threshold=True, nms_threshold=0.1,
        enable_samples=True, samples=8,
        use_label_channels=False,
        enable_order=False, order=5,
        device=None,
        output_basename="output",
):
    global default_model
    assert isinstance(img, np.ndarray), "Input must be a NumPy array."

    if device is None:
        if torch.cuda.device_count():
            device = 'cuda'
        else:
            device = 'cpu'
            
    meta = dict(
        cd_version=__version__,
        # filename=str(filename),
        model=model,
        device=device,
        use_label_channels=use_label_channels,
        enable_score_threshold=enable_score_threshold,
        score_threshold=float(score_threshold),
        enable_order=enable_order,
        order=order,
        enable_nms_threshold=enable_nms_threshold,
        nms_threshold=float(nms_threshold),
    )

    if model is None or len(str(model)) <= 0:
        model = default_model
    img = multi_norm(img, 'cstm-mix')  # TODO
    # Ensure the output directory exists (same as the .py file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    kw = {}
    if enable_score_threshold:
        kw['score_thresh'] = score_threshold
    if enable_nms_threshold:
        kw['nms_thresh'] = nms_threshold
    if enable_order:
        kw['order'] = order
    if enable_samples:
        kw['samples'] = samples
    m = CpnInterface(model.strip(), device=device, **kw)
    y = m(img, reduce_labels=not use_label_channels)
    dst_h5 = os.path.join(output_dir, os.path.basename(f"{output_basename}.h5"))
    to_h5(
        dst_h5, inputs=img, **y,
        attributes=dict(inputs=meta)
    )
    labels = y['labels']
    vis_labels = label_cmap(labels)
    dst_csv = os.path.join(output_dir, os.path.basename(f"{output_basename}.csv"))
    data.labels2property_table(
        labels,
        "label", "area", "feret_diameter_max", "bbox", "centroid", "convex_area",
        "eccentricity", "equivalent_diameter",
        "extent", "filled_area", "major_axis_length",
        "minor_axis_length", "orientation", "perimeter",
        "solidity", "mean_intensity", "max_intensity", "min_intensity",
        intensity_image=img
    ).to_csv(dst_csv)
    
    return vis_labels, img, dst_h5, dst_csv
    
class AIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Window")
        # Set a clean gradient background
        self.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
            font-family: 'Segoe UI', 'Arial', sans-serif;
        """)
        
        # Set initial window size and minimum size
        self.resize(1000, 700)  # More reasonable initial size
        self.setMinimumSize(800, 600)  # Keep minimum size reasonable
        
        # Create central widget and main layout with proper margins
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduced margins
        main_layout.setSpacing(10)  # Reduced spacing
        
        # Center buttons area (70% of width)
        center_widget = QWidget()
        center_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_col = QVBoxLayout(center_widget)
        center_col.setContentsMargins(15, 15, 15, 15)  # Consistent padding
        center_col.setSpacing(20)  # Slightly more spacing between buttons
        
        # Add scroll area for buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(200, 200, 200, 50);
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #7b3ff2;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(5, 5, 15, 5)  # Right margin for scrollbar
        button_layout.setSpacing(15)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Title label for the center section
        title_label = QLabel("AI Models")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0 15px 0;
            padding: 0;
        """)
        center_col.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Button container with scroll area for many buttons
        button_scroll = QScrollArea()
        button_scroll.setWidgetResizable(True)
        button_scroll.setFrameShape(QFrame.Shape.NoFrame)
        button_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0.1);
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(5, 5, 15, 5)  # Right margin for scrollbar
        button_layout.setSpacing(12)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Button style with relative sizing
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7e57c2, stop:1 #5e35b1);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 500;
                text-align: left;
                min-height: 60px;
                max-height: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9575cd, stop:1 #7e57c2);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5e35b1, stop:1 #4527a0);
            }
        """
        
        # Add buttons with icons and tooltips
        btns = [
            ("Kidney-Cyst-Stone Detection", "🔍", self.open_cv_kidney_detection),
            ("Kidney-Tumor-Cyst Segmentation", "🔬", self.open_cv_kidney_segmentation),
            ("Cancer Cell Segmentation", "🧫", self.open_cv_cell_segmentation),
            ("Kidney Disease Detection (Blood Test)", "💉", self.open_ml)
        ]
        
        for text, icon, func in btns:
            btn = QPushButton(f"{icon}  {text}")
            btn.setStyleSheet(f"""
                {button_style}
                QPushButton {{
                    padding: 12px 16px;
                    text-align: left;
                    word-wrap: break-word;
                    white-space: normal;
                }}
            """)
            font = QFont('Segoe UI', 12, QFont.Weight.Normal)
            btn.setFont(font)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            btn.clicked.connect(func)
            btn.setToolTip(text)
            button_layout.addWidget(btn)
            
        # Add stretch to push buttons to top
        button_layout.addStretch()
        
        # Set up scroll area
        button_scroll.setWidget(button_container)
        center_col.addWidget(button_scroll, 1)  # Take remaining space
        
        # Add center widget to main layout with stretch factor
        main_layout.addWidget(center_widget, 1)  # Center takes full width

    def open_cv_kidney_detection(self):
        self.setup_cv_kidney_detection_button()

    def open_cv_kidney_segmentation(self):
        self.setup_cv_kidney_segmentation_button()

    def open_cv_cell_segmentation(self):
        self.kidney_cell_segmentation_open_cv_link()

    def open_ml(self):
        self.open_ml_link()
        
    def open_ml_link(self):
        # Define column names
        columns = ['Bp','Sg','Al','Su','Rbc','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc','Htn']  # Assuming 14 input features
        # Define prediction function
        def predict(*args):
            input_data = pd.DataFrame([args], columns=columns)
            # Create a new row with values you want to normalize
            input_data = np.array([input_data])  # Replace with values you want to add to the new row
            # Load the CSV file into a DataFrame
            df = pd.read_csv(r"./ML_works/dataset/archive/new_model.csv")  # Assuming no column names in the CSV file
            df = df.iloc[:, 0:13].values    
            # Ensure the dimensions match by reshaping new_rows
            new_row = input_data.reshape(-1, df.shape[1])
            updated_array = np.concatenate((df, new_row), axis=0)
            # Perform min-max normalization
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(updated_array)
            # Get the last row of the array
            last_row = X_normalized[-1]
            # Ensure the dimensions match by reshaping new_rows
            input_data = last_row.reshape(-1, df.shape[1])          
            # print(input_data)
            prediction = kidney_ML_model.predict(input_data)
            # print(prediction)        
            if prediction[0] == 1:
                prediction_text = 'healthy'
            else:
                prediction_text = 'patient'           
            return prediction_text
        # Create Gradio interface
        inputs = []
        for col in columns:
            inputs.append(gr.Slider(label=col,minimum=0.000000, maximum=100000))
        output = gr.Textbox(label="Prediction")
        interface = gr.Interface(
            predict,
            inputs=inputs,
            outputs=output,
            title="Machine Learning Service for Kidney Disease",
            description="Enter values for the 13 input features to make a prediction.",
        )
        interface.launch(server_name= '0.0.0.0', server_port=8080, share=True)
        webbrowser.open_new_tab("http://localhost:8080/")

    def kidney_CV_model_pred(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = kidney_CV_model.predict(image, device='cpu', imgsz=(416,640), conf=0.1, iou=0.1)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
        return im_array
        
    def setup_cv_kidney_detection_button(self):
        interface = gr.Interface(
            self.kidney_CV_model_pred, 
            gr.Image(), 
            "image",
            title="Computer Vision Service for Kidney Disease Detection"
        )
        interface.launch(server_name='0.0.0.0', server_port=8081, share=True)
        webbrowser.open_new_tab("http://localhost:8081/")
            
    def setup_cv_kidney_segmentation_button(self):
        def kidney_cell_CV_model_pred(image, score_threshold, nms_threshold, samples, vertical_box=False):
            
            vis_labels, img, dst_h5, dst_csv = predict(image,
                                                       score_threshold=score_threshold, 
                                                       nms_threshold=nms_threshold,
                                                       samples=samples)
        
            # Convert vis_labels from boolean to uint8 and scale to [0, 255]
            vis_labels_uint8 = (vis_labels.astype(np.uint8)) * 255
    
            # Convert to BGR for colored visualization
            vis_labels_rgb_org = cv2.cvtColor(vis_labels_uint8, cv2.COLOR_GRAY2RGB)
            
            # Convert to BGR for colored visualization
            vis_labels_bgr = cv2.cvtColor(vis_labels_uint8, cv2.COLOR_GRAY2BGR)
    
            # Get image dimensions
            height, width = vis_labels_uint8.shape
    
            # Calculate overall metrics
            contours, _ = cv2.findContours(vis_labels_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            white_area = np.sum(vis_labels)  # True counts as 1
            black_area = vis_labels.size - white_area
            overall_ratio = int(black_area / white_area )if white_area > 0 else str('No cancer')
    
    
            if not vertical_box:
                # Define the central line and extended area
                center_y = height // 2
                top_line = max(0, center_y - 150)
                bottom_line = min(height, center_y + 150)
        
                # Extract the area between the two lines
                area_of_interest = vis_labels_uint8[top_line:bottom_line, :]
        
                # Find contours in the selected area
                contours_area, _ = cv2.findContours(area_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_count_area = len(contours_area)
        
                # Calculate black and white pixel counts in the selected area
                white_pixels_area = np.sum(area_of_interest > 0)
                black_pixels_area = area_of_interest.size - white_pixels_area
                black_white_ratio_area = int(black_pixels_area / white_pixels_area) if white_pixels_area > 0 else str('No cancer')
     
                # Find the nearest white pixels above and below the center line
                top_white_y = None
                bottom_white_y = None
                top_white_x = None
                bottom_white_x = None
                    
                # Search above the line
                for y in range(top_line - 1, -1, -1):  # Iterate upwards
                    if np.any(vis_labels_uint8[y, :] > 0):  # Check if any white pixel exists on this row
                        top_white_y = y
                        top_white_x = np.where(vis_labels_uint8[y, :] > 0)[0][0]
                        break
                
                # Search below the line
                for y in range(bottom_line + 1, height):  # Iterate downwards
                    if np.any(vis_labels_uint8[y, :] > 0):  # Check if any white pixel exists on this row
                        bottom_white_y = y
                        bottom_white_x = np.where(vis_labels_uint8[y, :] > 0)[0][0]
                        break
                
                # Calculate the vertical distance between the nearest white pixels above and below
                if top_white_y is not None and bottom_white_y is not None:
                    distance = bottom_white_y - top_white_y
                    cv2.circle(vis_labels_bgr, (top_white_x, top_white_y), 10, (0, 255, 255), -1) 
                    cv2.circle(vis_labels_bgr, (bottom_white_x, bottom_white_y), 10, (0, 255, 255), -1)  # Bottom white circle (yellow)# Top white circle (yellow)
                else:
                    distance = None  # No white area detected above or below
        
                # Draw the extended lines on the original image for visualization
                thickness = 3  # Line thickness
                cv2.line(vis_labels_bgr, (0, top_line), (width - 1, top_line), (0, 0, 255), thickness)  # Top line (red)
                cv2.line(vis_labels_bgr, (0, bottom_line - 1), (width - 1, bottom_line - 1), (0, 0, 255), thickness)  # Bottom line (red)
        
                # Add text to the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                color = (0, 0, 255)  # Red text
        
                # Add overall metrics to the top-left corner
                cv2.putText(vis_labels_bgr, f"Cancer cell in all region: {len(contours)}", (10, 30), font, font_scale, color, font_thickness)
                cv2.putText(vis_labels_bgr, f"Normal/Cancer area ratio in all region: {overall_ratio}", (10, 60), font, font_scale, color, font_thickness)
        
                right_text_x = max(width - 675, 10)  # Adjust to a minimum of 10 pixels from the left
               
                # Add area metrics to the top-right corner
                cv2.putText(vis_labels_bgr, f"Cancer cell in red region: {contour_count_area}", 
                            (right_text_x, 30), font, font_scale, color, font_thickness)
                cv2.putText(vis_labels_bgr, f"Normal/Cancer area ratio in red region: {black_white_ratio_area}", 
                            (right_text_x, 60), font, font_scale, color, font_thickness)
                # Display the distance on the image
                cv2.putText(vis_labels_bgr, f"Distance between two yellow blob: {distance}", 
                            (right_text_x, 90), font, font_scale, color, font_thickness)
                
                vis_labels_rgb = cv2.cvtColor(vis_labels_bgr, cv2.COLOR_BGR2RGB)
                
            if vertical_box:  
                
                # Define the central line and extended area
                center_x = width // 2
                left_line = max(0, center_x - 150)
                right_line = min(width, center_x + 150)
        
                # Extract the area between the two lines
                area_of_interest = vis_labels_uint8[:, left_line:right_line]
        
                # Find contours in the selected area
                contours_area, _ = cv2.findContours(area_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_count_area = len(contours_area)
        
                # Calculate black and white pixel counts in the selected area
                white_pixels_area = np.sum(area_of_interest > 0)
                black_pixels_area = area_of_interest.size - white_pixels_area
                black_white_ratio_area = int(black_pixels_area / white_pixels_area) if white_pixels_area > 0 else str('No cancer')
     
                # Initialize variables for left and right nearest white pixels
                left_white_x = None
                right_white_x = None
                left_white_y = None
                right_white_y = None
                
                # Search to the LEFT of the center line
                for x in range(left_line - 1, -1, -1):  # Iterate leftwards
                    if np.any(vis_labels_uint8[:, x] > 0):  # Check for white pixels in the column
                        left_white_x = x
                        left_white_y = np.where(vis_labels_uint8[:, x] > 0)[0][0]  # First white pixel's y-coordinate
                        break
                
                # Search to the RIGHT of the center line
                for x in range(right_line + 1, width):  # Iterate rightwards
                    if np.any(vis_labels_uint8[:, x] > 0):  # Check for white pixels in the column
                        right_white_x = x
                        right_white_y = np.where(vis_labels_uint8[:, x] > 0)[0][0]  # First white pixel's y-coordinate
                        break
                
                # Calculate the horizontal distance between the nearest white pixels
                if left_white_x is not None and right_white_x is not None:
                    distance = right_white_x - left_white_x
                    cv2.circle(vis_labels_bgr, (left_white_x, left_white_y), 10, (0, 255, 255), -1)  # Left white circle (yellow)
                    cv2.circle(vis_labels_bgr, (right_white_x, right_white_y), 10, (0, 255, 255), -1)  # Right white circle (yellow)
                else:
                    distance = None  # No white area detected to the left or right
        
                # Draw the extended lines on the original image for visualization
                thickness = 3  # Line thickness
                cv2.line(vis_labels_bgr, (left_line, 0), (left_line, height - 1), (0, 0, 255), thickness)
                cv2.line(vis_labels_bgr, (right_line, 0), (right_line, height - 1), (0, 0, 255), thickness)

                # Add text to the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                color = (0, 0, 255)  # Red text
        
                # Add overall metrics to the top-left corner
                cv2.putText(vis_labels_bgr, f"Cancer cell in all region: {len(contours)}", (10, 30), font, font_scale, color, font_thickness)
                cv2.putText(vis_labels_bgr, f"Normal/Cancer area ratio in all region: {overall_ratio}", (10, 60), font, font_scale, color, font_thickness)
        
                right_text_x = max(width - 675, 10)  # Adjust to a minimum of 10 pixels from the left
               
                # Add area metrics to the top-right corner
                cv2.putText(vis_labels_bgr, f"Cancer cell in red region: {contour_count_area}", 
                            (right_text_x, 30), font, font_scale, color, font_thickness)
                cv2.putText(vis_labels_bgr, f"Normal/Cancer area ratio in red region: {black_white_ratio_area}", 
                            (right_text_x, 60), font, font_scale, color, font_thickness)
                # Display the distance on the image
                cv2.putText(vis_labels_bgr, f"Distance between two yellow blob: {distance}", 
                            (right_text_x, 90), font, font_scale, color, font_thickness)
                
                vis_labels_rgb = cv2.cvtColor(vis_labels_bgr, cv2.COLOR_BGR2RGB)
                
            
            
            
            # Eşikleme
            _, thresh = cv2.threshold(vis_labels_uint8, 254, 255, cv2.THRESH_BINARY)

            # Maskeleri birleştirme
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=13)

            # Konturları bulma
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Minimum ve maksimum alan
            min_area = 20000
            max_area = 2300000  # Maksimum alan sınırı (örnek)

            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    if area > max_area:
                        result = cv2.cvtColor(vis_labels_uint8, cv2.COLOR_GRAY2BGR)  # Görüntüyü renklendir
                        warning_text = "Kanser Hucreleri %75'ten fazla kapanmistir!"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        color = (0, 0, 255)  # Kırmızı renk
                        thickness = 2
                        # text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[0]
                        # text_x = (result.shape[1] - text_size[0]) // 2  # Metni ortalamak için x koordinatı
                        # text_y = (result.shape[0] + text_size[1]) // 2  # Metni ortalamak için y koordinatı
                        cv2.putText(result, warning_text, (10, 30), font, font_scale, color, thickness)

                        # # Görüntüyü kaydet
                        # output_path = r"C:\Users\caglar.gurkan\Calismalar\DR\tez_calismalari\pemf\pemf_ai_cell\cancer_cell_image_ek_bilgiler_kodlar\TScratch\output_with_area_and_distance_mm2_v3.png"
                        # cv2.imwrite(output_path, result)

                        # print(f"Uyarı: Görüntü '{output_path}' konumuna kaydedildi.")
                    filtered_contours.append(contour)

            # Eğer 2'den az büyük kontur varsa görüntüye uyarı yaz
            if len(filtered_contours) < 2:
                result = cv2.cvtColor(vis_labels_uint8, cv2.COLOR_GRAY2BGR)  # Görüntüyü renklendir
                warning_text = "Kanser Hucreleri %75'ten fazla kapanmistir!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                color = (0, 0, 255)  # Kırmızı renk
                thickness = 2
                # text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[0]
                # text_x = (result.shape[1] - text_size[0]) // 2  # Metni ortalamak için x koordinatı
                # text_y = (result.shape[0] + text_size[1]) // 2  # Metni ortalamak için y koordinatı
                cv2.putText(result, warning_text, (10, 30), font, font_scale, color, thickness)

                # # Görüntüyü kaydet
                # output_path = r"C:\Users\caglar.gurkan\Calismalar\DR\tez_calismalari\pemf\pemf_ai_cell\cancer_cell_image_ek_bilgiler_kodlar\TScratch\output_with_area_and_distance_mm2_v3.png"
                # cv2.imwrite(output_path, result)

                # print(f"Uyarı: Görüntü '{output_path}' konumuna kaydedildi.")
            else:
                # En büyük iki konturu seç
                sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
                largest_contour = sorted_contours[0]
                second_largest_contour = sorted_contours[1]

                # İki konturun maskesini oluştur
                mask1 = np.zeros_like(vis_labels_uint8)
                mask2 = np.zeros_like(vis_labels_uint8)
                cv2.drawContours(mask1, [largest_contour], -1, 255, thickness=cv2.FILLED)
                cv2.drawContours(mask2, [second_largest_contour], -1, 255, thickness=cv2.FILLED)

                # İki kontur arasındaki alan
                combined_mask = cv2.bitwise_or(mask1, mask2)
                area_between = cv2.bitwise_and(dilated, cv2.bitwise_not(combined_mask))

                # Kalan alanın piksel sayısını hesapla
                remaining_area_pixels = np.sum(area_between == 255)

                # Pikseli bir birime dönüştürme (örneğin, 1 piksel² = 0.01 mm²)
                pixel_to_mm2 = 0.01  # Örneğin, 1 piksel² = 0.01 mm²
                remaining_area_mm2 = remaining_area_pixels * pixel_to_mm2

                # İki kontur arasındaki en kısa mesafeyi bulma
                distances = []
                for point1 in largest_contour:
                    for point2 in second_largest_contour:
                        distance = np.linalg.norm(point1[0] - point2[0])
                        distances.append((distance, tuple(point1[0]), tuple(point2[0])))

                # En kısa mesafeyi seç
                shortest_distance, point1, point2 = min(distances, key=lambda x: x[0])
                distance_mm = shortest_distance * pixel_to_mm2  # Pikselden mm'ye dönüştürme

                # Görüntüyü renklendirme
                result = cv2.cvtColor(vis_labels_uint8, cv2.COLOR_GRAY2BGR)

                # Konturları çiz
                cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)  # Yeşil çizgi
                cv2.drawContours(result, [second_largest_contour], -1, (255, 0, 0), 2)  # Mavi çizgi

                # En kısa mesafeyi kırmızı çizgi ile göster
                cv2.line(result, point1, point2, (0, 0, 255), 2)

                # Alan bilgisini görüntü üzerine yazma
                area_text = f"Area: {remaining_area_mm2:.2f} mm^2"
                distance_text = f"Distance: {distance_mm:.2f} mm"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                color = (0, 0, 255)  # Kırmızı renk
                thickness = 2
                cv2.putText(result, area_text, (10, 30), font, font_scale, color, thickness)
                cv2.putText(result, distance_text, (10, 60), font, font_scale, color, thickness)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            
            
            delete_after_process=True
            if delete_after_process:
                if os.path.exists(dst_h5):
                    os.remove(dst_h5)
                if os.path.exists(dst_csv):
                    os.remove(dst_csv)
                    
            return vis_labels_rgb_org, vis_labels_rgb, result
            
        interface = gr.Interface(kidney_cell_CV_model_pred, 
                                 inputs=[gr.Image(),
                                         gr.Slider(minimum=0, maximum=1.0, step=0.001, value=0.1, label="Score Threshold"),
                                         gr.Slider(minimum=0, maximum=1.0, step=0.001, value=0.1, label="NMS Threshold"),
                                         gr.Slider(minimum=8, maximum=256, step=1, value=8, label="Samples"),
                                         gr.Checkbox(label="Vertical Process for Cancer Cell", value=False),
                                         ],
                                 outputs= [gr.Image(), gr.Image(), gr.Image()], 
                                 title="Computer Vision Service for Kidney Cancer Cell Segmentation")
        
        
        interface.launch(server_name= '0.0.0.0', server_port=8089, share=True)
        webbrowser.open_new_tab("http://localhost:8089/")
                

        
class SensorDataWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Sensör sınıflarını içe aktar
        from sensors import HallSensor, TemperatureSensor, ECGSensor, CurrentSensor
        
        # Sensör nesnelerini oluştur
        self.hall_sensors = [HallSensor(i+1) for i in range(8)]
        self.temp_sensors = [TemperatureSensor(i+1) for i in range(8)]
        self.current_sensors = [CurrentSensor(i) for i in range(2)]
        self.ecg_sensor = ECGSensor()
        
        # Tüm sensörleri bir listede topla
        self.all_sensors = []
        self.all_sensors.extend(self.hall_sensors)
        self.all_sensors.extend(self.temp_sensors)
        self.all_sensors.extend(self.current_sensors)
        self.all_sensors.append(self.ecg_sensor)
        
        # Sensörlerin data_updated sinyallerini bağla
        for sensor in self.all_sensors:
            sensor.data_updated.connect(self.on_sensor_data_updated)
        
        self._running = True
        
        # Otomatik kaydetme için bayrak
        self.auto_save_enabled = True
        self.last_save_time = time.time()  # Son kaydetme zamanı
        self.save_interval = 1  # Her saniye kaydet (saniye cinsinden)
        
        # Hall sensör verileri için deque listeleri oluştur
        self.hall_data = [deque(maxlen=1500) for _ in range(8)]
        self.hall_time_data = [deque(maxlen=1500) for _ in range(8)]
        
        # Son veri değerlerini saklamak için sözlük
        self.last_data_values = {
            "hall": [0] * 8,
            "temp": {"ambient": [0] * 8, "object": [0] * 8},
            "current": [0] * 2,
            "ecg": 0
        }
        
        self.plot_update_interval = 0.01
        self.memory_cleanup_interval = 60
        self.last_memory_cleanup = time.time()
        # Önbellek işlemleri kaldırıldı
        self.start_time = time.time()  # Zaman takibi için başlangıç zamanı
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Veri kaydetme ve yükleme için durum mesajları
        self.save_status_label = QLabel("")
        self.status_bar.addWidget(self.save_status_label)

        self.udp_socket = None
        
        self.init_ui()
        
        self.udp_thread = threading.Thread(target=self._udp_thread_func, daemon=True)
        self.udp_thread.start()
        
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(int(self.plot_update_interval * 1000))
        
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._perform_memory_cleanup)
        self.memory_timer.start(10000)  # 10 saniyede bir kontrol et
        
        # Başlangıç durumunu göster
        self.update_status("ESP8266 sensör verilerini dinleniyor...")
    
    def _udp_thread_func(self):
        """UDP dinleyici thread fonksiyonu"""
        while self._running:
            self.udp_listener()
            time.sleep(0.001)  # CPU kullanımını azaltmak için kısa bekleme
    

    def init_ui(self):
        self.setWindowTitle("Sensör Veri Monitörü")
        self.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);         
            font-family: 'Segoe UI', 'Arial', sans-serif;
            color: #ffffff;
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a148c;
                color: white;
                border: 1px solid #7b1fa2;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #7b1fa2;
            }
            QPushButton:pressed {
                background-color: #9c27b0;
            }
            QCheckBox {
                color: #ffffff;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #7b1fa2;
                background-color: #2d185a;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #7b1fa2;
                background-color: #7b1fa2;
            }
            QGroupBox {
                border: 1px solid #7b1fa2;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)      
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Kontrol paneli ekleniyor
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.frameShape().NoFrame)
        
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(20, 10, 30, 20)
        self.plot_layout.setSpacing(20)
        
        self.create_hall_plots()
        self.create_current_plots()
        self.create_temp_plots()
        self.create_ecg_plot()
        
        self.plot_layout.addStretch()
        
        scroll.setWidget(self.plot_container)
        
        # Scroll widget'ını ana düzene ekle
        main_layout.addWidget(scroll)
        
    def create_control_panel(self):
        """Sensör kontrol panelini oluşturur"""
        # Sensör görünürlük durumlarını izlemek için sözlük
        self.sensor_visibility = {
            "hall": [True] * 8,
            "temp": [True] * 8,
            "current": [True] * 2,
            "ecg": True
        }
        
        # Grafik çizimi durumu
        self.plotting_active = True
        
        # Kontrol paneli için grup kutusu
        control_group = QGroupBox("Sensör Kontrol Paneli")
        control_layout = QVBoxLayout(control_group)
        
        # Üst kısım - Butonlar
        button_layout = QHBoxLayout()
        
        # Grafik çizimini durdur/başlat butonları
        self.stop_plotting_btn = QPushButton("Grafik Çizimini Durdur")
        self.stop_plotting_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.stop_plotting_btn.clicked.connect(self.toggle_plotting)
        self.stop_plotting_btn.setMinimumWidth(180)
        
        # Tüm sensörleri göster/gizle butonları
        self.show_all_btn = QPushButton("Tüm Sensörleri Göster")
        self.show_all_btn.clicked.connect(self.show_all_sensors)
        self.show_all_btn.setMinimumWidth(150)
        
        self.hide_all_btn = QPushButton("Tüm Sensörleri Gizle")
        self.hide_all_btn.clicked.connect(self.hide_all_sensors)
        self.hide_all_btn.setMinimumWidth(150)
        
        # Butonları düzene ekle
        button_layout.addWidget(self.stop_plotting_btn)
        button_layout.addWidget(self.show_all_btn)
        button_layout.addWidget(self.hide_all_btn)
        button_layout.addStretch()
        
        control_layout.addLayout(button_layout)
        
        # Alt kısım - Sensör kontrolleri
        sensor_controls_layout = QHBoxLayout()
        
        # Hall sensör kontrolleri
        hall_group = QGroupBox("Hall Sensörleri")
        hall_layout = QVBoxLayout(hall_group)
        self.hall_checkboxes = []
        
        for i in range(8):
            cb = QCheckBox(f"Coil {i+1} Hall Sensörü")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, idx=i: self.toggle_hall_sensor(idx, state))
            hall_layout.addWidget(cb)
            self.hall_checkboxes.append(cb)
        
        # Sıcaklık sensör kontrolleri
        temp_group = QGroupBox("Sıcaklık Sensörleri")
        temp_layout = QVBoxLayout(temp_group)
        self.temp_checkboxes = []
        
        for i in range(8):
            cb = QCheckBox(f"Coil {i+1} Sıcaklık Sensörü")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, idx=i: self.toggle_temp_sensor(idx, state))
            temp_layout.addWidget(cb)
            self.temp_checkboxes.append(cb)
        
        # Akım ve EKG sensör kontrolleri
        other_group = QGroupBox("Diğer Sensörler")
        other_layout = QVBoxLayout(other_group)
        
        # Akım sensörleri
        self.current_checkboxes = []
        for i in range(2):
            cb = QCheckBox(f"Akım Sensörü {i+1}")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, idx=i: self.toggle_current_sensor(idx, state))
            other_layout.addWidget(cb)
            self.current_checkboxes.append(cb)
        
        # EKG sensörü
        self.ecg_checkbox = QCheckBox("EKG Sensörü")
        self.ecg_checkbox.setChecked(True)
        self.ecg_checkbox.stateChanged.connect(self.toggle_ecg_sensor)
        other_layout.addWidget(self.ecg_checkbox)
        
        # Sensör gruplarını düzene ekle
        sensor_controls_layout.addWidget(hall_group)
        sensor_controls_layout.addWidget(temp_group)
        sensor_controls_layout.addWidget(other_group)
        
        control_layout.addLayout(sensor_controls_layout)
        
        return control_group

    def save_sensor_data(self):
        """Tüm sensör verilerini Excel dosyalarına kaydet"""
        try:
            # Excel formatında kaydet
            selected_format = "excel"
            
            # EKG verilerini kaydet
            ecg_filename = self.ecg_sensor.save_data_to_file()
            
            # Hall sensör verilerini kaydet
            hall_filenames = []
            for i, sensor in enumerate(self.hall_sensors):
                if len(sensor.data) > 0:  # Veri varsa kaydet
                    hall_filenames.append(sensor.save_data_to_file())
            
            # Sıcaklık sensör verilerini kaydet
            temp_filenames = []
            for i, sensor in enumerate(self.temp_sensors):
                if len(sensor.object_data) > 0:  # Veri varsa kaydet
                    temp_filenames.append(sensor.save_data_to_file())
            
            # Akım sensör verilerini kaydet
            current_filenames = []
            for i, sensor in enumerate(self.current_sensors):
                if len(sensor.data) > 0:  # Veri varsa kaydet
                    current_filenames.append(sensor.save_data_to_file())
            
            # Durum mesajını güncelle
            saved_files = []
            if ecg_filename:
                saved_files.append(f"EKG: {os.path.basename(ecg_filename)}")
            if hall_filenames:
                saved_files.append(f"Hall: {len(hall_filenames)} dosya")
            if temp_filenames:
                saved_files.append(f"Sıcaklık: {len(temp_filenames)} dosya")
            if current_filenames:
                saved_files.append(f"Akım: {len(current_filenames)} dosya")
                
            if saved_files:
                self.save_status_label.setText(f"Kaydedilen dosyalar (EXCEL): {', '.join(saved_files)}")
            else:
                self.save_status_label.setText("Kaydedilecek veri bulunamadı.")
                
        except Exception as e:
            self.save_status_label.setText(f"Hata: {str(e)}")
            print(f"Veri kaydedilirken hata oluştu: {e}")
    
    def load_sensor_data(self):
        """Sensör verilerini dosyadan yükle"""
        try:
            # Dosya seçme diyaloğunu göster
            filename, _ = QFileDialog.getOpenFileName(
                self, "Sensör Verisi Yükle", "", "Excel Dosyaları (*.xlsx *.xls)"
            )
            
            if not filename:
                return
            
            # Dosya adından sensör tipini belirle
            loaded = False
            
            # EKG sensörü için kontrol et
            if "ECG" in filename or "ecg" in filename:
                if self.ecg_sensor.load_data_from_file(filename):
                    self.save_status_label.setText(f"EKG verileri yüklendi: {os.path.basename(filename)}")
                    loaded = True
            
            # Hall sensörleri için kontrol et
            for i, sensor in enumerate(self.hall_sensors):
                if f"Hall_Sensörü_(Coil_{i+1})" in filename:
                    if sensor.load_data_from_file(filename):
                        self.save_status_label.setText(f"Hall sensör {i+1} verileri yüklendi: {os.path.basename(filename)}")
                        loaded = True
                        break
            
            # Sıcaklık sensörleri için kontrol et
            for i, sensor in enumerate(self.temp_sensors):
                if f"Sıcaklık_Sensörü_(Coil_{i+1})" in filename:
                    if sensor.load_data_from_file(filename):
                        self.save_status_label.setText(f"Sıcaklık sensör {i+1} verileri yüklendi: {os.path.basename(filename)}")
                        loaded = True
                        break
            
            # Akım sensörleri için kontrol et
            for i, sensor in enumerate(self.current_sensors):
                if f"Akım_Sensörü_{i+1}" in filename:
                    if sensor.load_data_from_file(filename):
                        self.save_status_label.setText(f"Akım sensör {i+1} verileri yüklendi: {os.path.basename(filename)}")
                        loaded = True
                        break
            
            if not loaded:
                self.save_status_label.setText(f"Dosya yüklenemedi veya uyumsuz format: {os.path.basename(filename)}")
            
            # Grafikleri güncelle
            self.update_plots()
            
        except Exception as e:
            self.save_status_label.setText(f"Hata: {str(e)}")
            print(f"Veri yüklenirken hata oluştu: {e}")

    def create_hall_plots(self):
        group = QGroupBox("ESP8266 Hall Effect Sensors")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        group.setMinimumHeight(800)
        
        # Create a grid layout for the plots (2x4 grid for 8 hall effect graphs)
        grid = QGridLayout(group)
        grid.setSpacing(10)
        
        # Create 8 hall effect plots (2 rows, 4 columns)
        self.hall_plots = []
        for i in range(8):
            row = i // 4
            col = i % 4
            
            # Create a container for each plot with a border
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground(None)
            
            # Style the plot
            plot_widget.setLabel('left', f'Manyetik Alan (mT)', color='white')
            plot_widget.setLabel('bottom', 'Zaman (s)', color='white')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Store plot reference for later use
            self.hall_plots.append(plot_widget.getPlotItem())
            
            # Highlight Coil ID 1 with special title
            if i == 0:  # Coil ID 1
                plot_widget.setTitle(f'Coil {i+1} - Hall Sensörü ', color='#00ff7f')
                # Add a note about real-time data
                note_label = QLabel("ESP8266 Gerçek Zamanlı Veri")
                note_label.setStyleSheet("color: #00ff7f; font-style: italic;")
                note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(note_label, row+1, col)
            else:
                plot_widget.setTitle(f'Coil {i+1} - Hall Sensörü', color='white')
            
            # Add plot to the grid
            grid.addWidget(plot_widget, row, col)
            
            # Create and store the plot curve
            # Use a brighter color for Coil ID 1
            if i == 0:  # Coil ID 1
                pen = pg.mkPen(color='#00ff7f', width=3)  # Bright green, thicker line
            else:
                pen = pg.mkPen(color=pg.intColor(i, 8, alpha=200), width=2)
                
            curve = plot_widget.plot(pen=pen)
            # Sensör nesnesine eğriyi ayarla
            self.hall_sensors[i].set_plot_curve(curve)
        
        self.plot_layout.addWidget(group)
        
    def create_current_plots(self):
        group = QGroupBox("ESP32 Akım Sensörleri")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        group.setMinimumHeight(400)
        
        # Create a grid layout for the plots (1x2 grid for 2 current sensor graphs)
        grid = QGridLayout(group)
        grid.setSpacing(10)
        
        # Create 2 current sensor plots (1 row, 2 columns)
        self.current_plots = []
        
        sensor_names = ["Akım Sensörü 1", "Akım Sensörü 2"]
        
        for i in range(2):
            # Create a container for each plot with a border
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground(None)
            
            # Style the plot
            plot_widget.setLabel('left', f'Akım (A)', color='white')
            plot_widget.setLabel('bottom', 'Zaman (s)', color='white')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setTitle(f'{sensor_names[i]}', color='white')
            
            # Store plot reference for later use
            self.current_plots.append(plot_widget.getPlotItem())
            
            # Add plot to the grid
            grid.addWidget(plot_widget, 0, i)
            
            # Create and store the plot curve
            pen = pg.mkPen(color=pg.intColor(i+10, 8, alpha=200), width=2)
            curve = plot_widget.plot(pen=pen)
            # Sensör nesnesine eğriyi ayarla
            self.current_sensors[i].set_plot_curve(curve)
        
        self.plot_layout.addWidget(group)

    def create_ecg_plot(self):
        group = QGroupBox("ESP32 EKG Sensörü")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        group.setMinimumHeight(400)
        group.setStyleSheet("QGroupBox { border: 2px solid #00FF00; border-radius: 5px; margin-top: 1ex; } "
                          "QGroupBox::title { color: #00FF00; subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }")
        
        # Create a layout for the ECG plot
        layout = QVBoxLayout(group)
        
        # Create ECG plot
        plot_widget = pg.PlotWidget()
        
        # Hastane EKG monitörü görünümü için arka plan rengini ayarla
        plot_widget.setBackground('#000000')  # Siyah arka plan
        
        # Izgara çizgilerini hastane EKG cihazı gibi ayarla
        plot_widget.showGrid(x=True, y=True, alpha=0.7)
        plot_widget.getAxis('left').setPen(pg.mkPen(color='#00FF00', width=1))  # Yeşil eksen
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#00FF00', width=1))  # Yeşil eksen
        
        # Izgara çizgilerini yeşil yap
        plot_widget.getAxis('left').setGrid(255)  # Izgara çizgilerini göster
        plot_widget.getAxis('bottom').setGrid(255)  # Izgara çizgilerini göster
        plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#00FF00'))  # Yeşil yazı
        plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#00FF00'))  # Yeşil yazı
        
        # Izgara çizgilerinin rengini ayarla
        plot_widget.getPlotItem().getViewBox().setBackgroundColor('#000000')  # Siyah arka plan
        
        # Hastane EKG cihazlarındaki gibi etiketler
        plot_widget.setLabel('left', 'mV', color='#00FF00')
        plot_widget.setLabel('bottom', 'Zaman (s)', color='#00FF00')
        plot_widget.setTitle('EKG Monitörü', color='#00FF00')
        
        # Hastane monitörü görünümü için ızgara çizgilerini özelleştir
        grid_pen = pg.mkPen(color='#00AA00', width=1, style=Qt.PenStyle.DotLine)
        plot_widget.getPlotItem().getViewBox().setBackgroundColor('#000000')  # Siyah arka plan
        
        # Kalp atış hızı için etiket ekle
        self.heart_rate_label = QLabel("Kalp Atış Hızı: -- BPM")
        self.heart_rate_label.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")
        self.heart_rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.heart_rate_label)
        
        # Elektrot bağlantı durumu için etiket ekle
        self.leads_status_label = QLabel("Elektrot Durumu: Bilinmiyor")
        self.leads_status_label.setStyleSheet("color: #FFFF00; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")
        self.leads_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.leads_status_label)
        
        # Vital değerler paneli kaldırıldı
        
        # Store plot reference for later use
        self.ecg_plot = plot_widget.getPlotItem()
        
        # Add plot to the layout
        layout.addWidget(plot_widget)
        
        # Sadece filtrelenmiş veri için eğri oluştur
        # Filtrelenmiş veri için yeşil eğri (kalın çizgi)
        curve = plot_widget.plot(pen=pg.mkPen(color='#00FF00', width=2))
        # Sensör nesnesine eğriyi ayarla
        self.ecg_sensor.set_plot_curve(curve)
        
        # İzoelektrik çizgi (0 mV seviyesi)
        self.iso_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color='#FFFFFF', width=1, style=Qt.PenStyle.DotLine))
        plot_widget.addItem(self.iso_line)
        
        # Milimetre kağıdı görünümü için arka plan ızgarası ekle
        for i in range(-10, 11):
            # Yatay çizgiler
            h_line = pg.InfiniteLine(pos=i, angle=0, pen=pg.mkPen(color='#004400', width=1, style=Qt.PenStyle.DotLine))
            plot_widget.addItem(h_line)
            # Dikey çizgiler
            if i >= 0 and i <= 10:
                v_line = pg.InfiniteLine(pos=i, angle=90, pen=pg.mkPen(color='#004400', width=1, style=Qt.PenStyle.DotLine))
                plot_widget.addItem(v_line)
        
        self.plot_layout.addWidget(group)
    
    def create_temp_plots(self):
        group = QGroupBox("ESP8266 Sıcaklık Sensörleri")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        group.setMinimumHeight(800)
        
        # Create a grid layout for the plots (2x4 grid for 8 temperature graphs)
        grid = QGridLayout(group)
        grid.setSpacing(10)
        
        # Create 8 temperature plots (2 rows, 4 columns)
        self.temp_plots = []
        for i in range(8):
            row = i // 4
            col = i % 4
            
            # Create a container for each plot with a border
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground(None)
            
            # Style the plot
            plot_widget.setLabel('left', f'Sıcaklık (°C)', color='white')
            plot_widget.setLabel('bottom', 'Zaman (s)', color='white')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Store plot reference for later use
            self.temp_plots.append(plot_widget.getPlotItem())
            
            # Highlight Coil ID 1 with special title
            if i == 0:  # Coil ID 1
                plot_widget.setTitle(f'Coil {i+1} - Sıcaklık Sensörleri ', color='#00ff7f')
                # Add a note about real-time data
                note_label = QLabel("ESP8266 Gerçek Zamanlı Veri")
                note_label.setStyleSheet("color: #00ff7f; font-style: italic;")
                note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(note_label, row+1, col)
            else:
                plot_widget.setTitle(f'Coil {i+1} - Sıcaklık Sensörleri', color='white')
                
            plot_widget.addLegend()
            
            # Add plot to the grid
            grid.addWidget(plot_widget, row, col)
            
            # Create and store the plot curves (one for object, one for ambient)
            if i == 0:  # Coil ID 1
                object_pen = pg.mkPen(color='#ff5500', width=3)  # Brighter orange for object
                ambient_pen = pg.mkPen(color='#00aaff', width=3)  # Brighter blue for ambient
            else:
                object_pen = pg.mkPen(color='#ff7f0e', width=2)  # Orange for object
                ambient_pen = pg.mkPen(color='#1f77b4', width=2)  # Blue for ambient
            
            object_curve = plot_widget.plot(pen=object_pen, name='Nesne')
            ambient_curve = plot_widget.plot(pen=ambient_pen, name='Ortam')
            
            # Sensör nesnesine eğrileri ayarla
            self.temp_sensors[i].set_plot_curves(object_curve, ambient_curve)
        
        self.plot_layout.addWidget(group)

    # ESP32 ile uyumlu olması için ECG ile ilgili fonksiyon kaldırıldı
    # def create_ecg_plot(self):
    #     pass
        
    def udp_listener(self):
        """ESP8266 ve ESP32 ile uyumlu UDP dinleyici"""
        try:
            # UDP soketini oluştur ve bağla
            if not hasattr(self, 'udp_socket') or self.udp_socket is None:
                self._create_udp_socket()
            elif hasattr(self, 'connection_lost') and self.connection_lost:
                # Bağlantı kaybı durumunda yeniden bağlanma girişimi
                self._reconnect_udp()
            
            # UDP verilerini al
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                # Bağlantı başarılı olduğunda bağlantı durumunu sıfırla
                if hasattr(self, 'connection_lost') and self.connection_lost:
                    self.connection_lost = False
                    self.connection_retry_count = 0
                    self.update_status("UDP bağlantısı yeniden kuruldu, sensör verilerini dinleniyor...")
                
                json_data = json.loads(data.decode('utf-8'))
                
                # Debug için JSON verisini göster
                print(f"Alınan veri: {json_data}")
                
                # Coil ID'yi al
                coil_id = int(json_data.get('coil_id', 0))
                sensor_name = json_data.get('sensor_name', '')
                
                # Akım sensörü verilerini işle
                if 'current' in json_data and sensor_name in ["Akım Sensörü 1", "Akım Sensörü 2"]:
                    current_value = float(json_data['current'])
                    current_time = time.time() - self.start_time
                    
                    # Hangi akım sensörü olduğunu belirle
                    sensor_index = 0 if sensor_name == "Akım Sensörü 1" else 1
                    
                    # Sensör nesnesini kullanarak veriyi işle
                    self.current_sensors[sensor_index].process_data(current_value, current_time)
                    
                    # WiFi sinyal gücü ve diğer bilgileri durum çubuğunda göster
                    if 'wifi_rssi' in json_data:
                        rssi = json_data['wifi_rssi']
                        self.update_status(f"{sensor_name} bağlı - WiFi: {rssi} dBm")
                
                # EKG sensörü verilerini işle
                elif 'ecg_value' in json_data and sensor_name == "ECG Sensörü":
                    ecg_value = float(json_data['ecg_value'])
                    current_time = time.time() - self.start_time
                    
                    # Sensör nesnesini kullanarak veriyi işle
                    self.ecg_sensor.process_data(ecg_value, current_time)
                    
                    # Debug için veri ekleme bilgisini göster
                    print(f"EKG veri eklendi: Ham={ecg_value:.3f}V, zaman: {current_time:.2f}s")
                    
                    # Elektrot bağlantı durumunu kontrol et ve göster
                    if 'leads_connected' in json_data:
                        leads_connected = json_data['leads_connected']
                        if hasattr(self, 'leads_status_label'):
                            if leads_connected:
                                self.leads_status_label.setText("Elektrot Durumu: BAĞLI ✓")
                                self.leads_status_label.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")  # Yeşil
                            else:
                                self.leads_status_label.setText("Elektrot Durumu: BAĞLI DEĞİL ✗")
                                self.leads_status_label.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")  # Kırmızı
                                # Elektrotlar bağlı değilse durum çubuğunda uyarı göster
                                self.update_status("UYARI: EKG elektrotları bağlı değil! Lütfen elektrot bağlantılarını kontrol edin.")
                    
                    # Kalp atış hızı bilgisini göster
                    if 'heart_rate' in json_data:
                        heart_rate = json_data['heart_rate']
                        # Durum çubuğunda göster
                        self.update_status(f"EKG Sensörü bağlı - Kalp Atış Hızı: {heart_rate} BPM")
                        # EKG grafiğinin altındaki etikette göster
                        if hasattr(self, 'heart_rate_label'):
                            self.heart_rate_label.setText(f"Kalp Atış Hızı: {heart_rate} BPM")
                            # Kalp atış hızına göre renk değiştir
                            if heart_rate < 60:  # Bradikardi
                                self.heart_rate_label.setStyleSheet("color: #00BFFF; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")  # Mavi
                                # EKG eğrisinin rengini de değiştir
                                if hasattr(self, 'ecg_curve'):
                                    self.ecg_curve.setPen(pg.mkPen(color='#00BFFF', width=2))
                            elif heart_rate > 100:  # Taşikardi
                                self.heart_rate_label.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")  # Kırmızı
                                # EKG eğrisinin rengini de değiştir
                                if hasattr(self, 'ecg_curve'):
                                    self.ecg_curve.setPen(pg.mkPen(color='#FF0000', width=2))
                            else:  # Normal
                                self.heart_rate_label.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold; background-color: #000000; padding: 5px; border: 1px solid #00AA00;")  # Yeşil
                                # EKG eğrisinin rengini de değiştir
                                if hasattr(self, 'ecg_curve'):
                                    self.ecg_curve.setPen(pg.mkPen(color='#00FF00', width=2))
                
                # ESP8266 Hall ve sıcaklık sensörü verilerini işle
                elif 1 <= coil_id <= 8:
                    # Manyetik alan verisi
                    if 'magnetic_magnitude' in json_data:
                        mag_value = float(json_data['magnetic_magnitude'])
                        current_time = time.time() - self.start_time
                        
                        # Hall sensör nesnesini kullanarak veriyi işle
                        self.hall_sensors[coil_id-1].process_data({'hall_value': mag_value, 'coil_id': coil_id})
                        
                        # Eski kod referansı için tutuluyor (kaldırılabilir)
                        self.hall_data[coil_id-1].append(mag_value)
                        self.hall_time_data[coil_id-1].append(current_time)
                        self.last_data_values["hall"][coil_id-1] = mag_value
                        
                        # Debug için veri ekleme bilgisini göster
                        print(f"Hall veri eklendi - Coil {coil_id}: {mag_value} mT, zaman: {current_time:.2f}s")
                    
                    # Sıcaklık verileri
                    if 'ambient_temp' in json_data and 'object_temp' in json_data:
                        ambient_temp = float(json_data['ambient_temp'])
                        object_temp = float(json_data['object_temp'])
                        current_time = time.time() - self.start_time
                        
                        # Sensör nesnesini kullanarak veriyi işle
                        self.temp_sensors[coil_id-1].process_data(ambient_temp, object_temp, current_time)
                        
                        # Debug için veri ekleme bilgisini göster
                        print(f"Sıcaklık veri eklendi - Coil {coil_id}: Ambient: {ambient_temp}°C, Object: {object_temp}°C, zaman: {current_time:.2f}s")
                    
                    # WiFi sinyal gücü ve diğer bilgileri durum çubuğunda göster
                    if 'rssi' in json_data:
                        rssi = json_data['rssi']
                        self.update_status(f"Coil {coil_id} bağlı - WiFi: {rssi} dBm")
                else:
                    self.update_status(f"Bilinmeyen sensör verisi alındı: {sensor_name} (Coil ID: {coil_id})")
                    
            except BlockingIOError:
                # Veri yoksa devam et
                pass
            except json.JSONDecodeError:
                self.update_status("Hatalı JSON verisi alındı")
            except ConnectionResetError:
                # Bağlantı sıfırlandı hatası
                self._handle_connection_loss("Bağlantı sıfırlandı")
            except ConnectionRefusedError:
                # Bağlantı reddedildi hatası
                self._handle_connection_loss("Bağlantı reddedildi")
            except Exception as e:
                self.update_status(f"Veri alımında hata: {str(e)}")
        except Exception as e:
            self.update_status(f"UDP dinleyici hatası: {str(e)}")
            traceback.print_exc()
            # Kritik hata durumunda bağlantı kaybı olarak işaretle
            self._handle_connection_loss(str(e))
    
    def _create_udp_socket(self):
        """UDP soketini oluştur ve bağla"""
        try:
            if hasattr(self, 'udp_socket') and self.udp_socket:
                try:
                    self.udp_socket.close()
                except:
                    pass
            
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind(('0.0.0.0', 4210))  # ESP8266'dan gelen verileri dinle
            self.udp_socket.setblocking(False)
            self.connection_lost = False
            self.connection_retry_count = 0
            self.update_status("UDP bağlantısı kuruldu, sensör verilerini dinleniyor...")
            return True
        except Exception as e:
            self.update_status(f"UDP soketi oluşturma hatası: {str(e)}")
            return False
    
    def _handle_connection_loss(self, error_msg):
        """Bağlantı kaybı durumunu işle"""
        if not hasattr(self, 'connection_lost'):
            self.connection_lost = False
            self.connection_retry_count = 0
            self.last_connection_attempt = 0
        
        # Bağlantı kaybı durumunu işaretle
        if not self.connection_lost:
            self.connection_lost = True
            self.update_status(f"UDP bağlantısı kesildi: {error_msg}. Yeniden bağlanmaya çalışılacak...")
            
            # Kritik hata durumunda kullanıcıya bildirim göster
            if self.connection_retry_count >= 5:
                self._show_critical_error(f"UDP bağlantısı kesildi: {error_msg}\n\nYeniden bağlanmaya çalışılıyor...")
    
    def _reconnect_udp(self):
        """UDP bağlantısını yeniden kurma girişimi"""
        current_time = time.time()
        
        # Son bağlantı girişiminden en az 5 saniye geçtiyse yeniden dene
        if not hasattr(self, 'last_connection_attempt') or current_time - self.last_connection_attempt > 5:
            self.last_connection_attempt = current_time
            self.connection_retry_count += 1
            
            self.update_status(f"UDP bağlantısı yeniden kuruluyor... (Deneme {self.connection_retry_count})")
            success = self._create_udp_socket()
            
            if success:
                self.connection_lost = False
                self.connection_retry_count = 0
                self.update_status("UDP bağlantısı başarıyla yeniden kuruldu.")
            elif self.connection_retry_count >= 10:
                # 10 başarısız denemeden sonra kritik hata göster
                self._show_critical_error("UDP bağlantısı kurulamadı. Uygulama yeniden başlatılacak.")
                self._restart_application()
    
    def _show_critical_error(self, message):
        """Kritik hata durumunda kullanıcıya bildirim göster"""
        # Ana thread'de çalıştırmak için QTimer kullan
        QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Kritik Bağlantı Hatası", message))
    
    def _restart_application(self):
        """Uygulamayı kontrollü şekilde yeniden başlat"""
        # Ana thread'de çalıştırmak için QTimer kullan
        QTimer.singleShot(0, lambda: self.close())
        QTimer.singleShot(100, lambda: os.execl(sys.executable, sys.executable, *sys.argv))

    def update_plots(self):
        """Grafikleri güncelleme"""
        try:
            # Grafik çizimi aktif değilse güncelleme yapma
            if not hasattr(self, 'plotting_active') or not self.plotting_active:
                return
                
            # Akım sensörlerini güncelle
            for i, sensor in enumerate(self.current_sensors):
                if hasattr(self, 'sensor_visibility') and self.sensor_visibility["current"][i]:
                    sensor.update_plot()
            
            # EKG sensörünü güncelle
            if hasattr(self, 'sensor_visibility') and self.sensor_visibility["ecg"]:
                self.ecg_sensor.update_plot()
            
            # Hall effect sensörlerini güncelle
            for i, sensor in enumerate(self.hall_sensors):
                if hasattr(self, 'sensor_visibility') and self.sensor_visibility["hall"][i]:
                    sensor.update_plot()
            
            # Sıcaklık sensörlerini güncelle
            for i, sensor in enumerate(self.temp_sensors):
                if hasattr(self, 'sensor_visibility') and self.sensor_visibility["temp"][i]:
                    sensor.update_plot()
        except Exception as e:
            self.update_status(f"Grafik güncelleme hatası: {str(e)}")
            traceback.print_exc()

    # Sensör görünürlük kontrol fonksiyonları
    def toggle_hall_sensor(self, index, state):
        """Hall sensörünün görünürlüğünü değiştirir"""
        self.sensor_visibility["hall"][index] = (state == Qt.CheckState.Checked.value)
        # Sensör grafiğini gizle/göster
        if hasattr(self, 'hall_plots') and index < len(self.hall_plots):
            plot_widget = self.hall_plots[index].getViewWidget()
            plot_widget.setVisible(self.sensor_visibility["hall"][index])
            # Görünürlük değiştiğinde grafik verilerini temizle ve yeniden çiz
            if self.sensor_visibility["hall"][index]:
                # Grafiği temizle ve yeniden çiz
                self.hall_plots[index].clear()
                if hasattr(self, 'hall_sensors') and index < len(self.hall_sensors):
                    self.hall_sensors[index].update_plot()
    
    def toggle_temp_sensor(self, index, state):
        """Sıcaklık sensörünün görünürlüğünü değiştirir"""
        self.sensor_visibility["temp"][index] = (state == Qt.CheckState.Checked.value)
        # Sensör grafiğini gizle/göster
        if hasattr(self, 'temp_plots') and index < len(self.temp_plots):
            plot_widget = self.temp_plots[index].getViewWidget()
            plot_widget.setVisible(self.sensor_visibility["temp"][index])
            # Görünürlük değiştiğinde grafik verilerini temizle ve yeniden çiz
            if self.sensor_visibility["temp"][index]:
                # Grafiği temizle ve yeniden çiz
                self.temp_plots[index].clear()
                if hasattr(self, 'temp_sensors') and index < len(self.temp_sensors):
                    self.temp_sensors[index].update_plot()
    
    def toggle_current_sensor(self, index, state):
        """Akım sensörünün görünürlüğünü değiştirir"""
        self.sensor_visibility["current"][index] = (state == Qt.CheckState.Checked.value)
        # Sensör grafiğini gizle/göster
        if hasattr(self, 'current_plots') and index < len(self.current_plots):
            plot_widget = self.current_plots[index].getViewWidget()
            plot_widget.setVisible(self.sensor_visibility["current"][index])
            # Görünürlük değiştiğinde grafik verilerini temizle ve yeniden çiz
            if self.sensor_visibility["current"][index]:
                # Grafiği temizle ve yeniden çiz
                self.current_plots[index].clear()
                if hasattr(self, 'current_sensors') and index < len(self.current_sensors):
                    self.current_sensors[index].update_plot()
    
    def toggle_ecg_sensor(self, state):
        """EKG sensörünün görünürlüğünü değiştirir"""
        self.sensor_visibility["ecg"] = (state == Qt.CheckState.Checked.value)
        # Sensör grafiğini gizle/göster
        if hasattr(self, 'ecg_plot'):
            plot_widget = self.ecg_plot.getViewWidget()
            plot_widget.setVisible(self.sensor_visibility["ecg"])
    
    def show_all_sensors(self):
        """Tüm sensörleri gösterir"""
        # Hall sensörleri
        for i in range(8):
            self.sensor_visibility["hall"][i] = True
            self.hall_checkboxes[i].setChecked(True)
            if hasattr(self, 'hall_plots') and i < len(self.hall_plots):
                plot_widget = self.hall_plots[i].getViewWidget()
                plot_widget.setVisible(True)
                # Grafiği temizle ve yeniden çiz
                self.hall_plots[i].clear()
                if hasattr(self, 'hall_sensors') and i < len(self.hall_sensors):
                    self.hall_sensors[i].update_plot()
        
        # Sıcaklık sensörleri
        for i in range(8):
            self.sensor_visibility["temp"][i] = True
            self.temp_checkboxes[i].setChecked(True)
            if hasattr(self, 'temp_plots') and i < len(self.temp_plots):
                plot_widget = self.temp_plots[i].getViewWidget()
                plot_widget.setVisible(True)
                # Grafiği temizle ve yeniden çiz
                self.temp_plots[i].clear()
                if hasattr(self, 'temp_sensors') and i < len(self.temp_sensors):
                    self.temp_sensors[i].update_plot()
        
        # Akım sensörleri
        for i in range(2):
            self.sensor_visibility["current"][i] = True
            self.current_checkboxes[i].setChecked(True)
            if hasattr(self, 'current_plots') and i < len(self.current_plots):
                plot_widget = self.current_plots[i].getViewWidget()
                plot_widget.setVisible(True)
                # Grafiği temizle ve yeniden çiz
                self.current_plots[i].clear()
                if hasattr(self, 'current_sensors') and i < len(self.current_sensors):
                    self.current_sensors[i].update_plot()
        
        # EKG sensörü
        self.sensor_visibility["ecg"] = True
        self.ecg_checkbox.setChecked(True)
        if hasattr(self, 'ecg_plot'):
            plot_widget = self.ecg_plot.getViewWidget()
            plot_widget.setVisible(True)
    
    def hide_all_sensors(self):
        """Tüm sensörleri gizler"""
        # Hall sensörleri
        for i in range(8):
            self.sensor_visibility["hall"][i] = False
            self.hall_checkboxes[i].setChecked(False)
            if hasattr(self, 'hall_plots') and i < len(self.hall_plots):
                plot_widget = self.hall_plots[i].getViewWidget()
                plot_widget.setVisible(False)
        
        # Sıcaklık sensörleri
        for i in range(8):
            self.sensor_visibility["temp"][i] = False
            self.temp_checkboxes[i].setChecked(False)
            if hasattr(self, 'temp_plots') and i < len(self.temp_plots):
                plot_widget = self.temp_plots[i].getViewWidget()
                plot_widget.setVisible(False)
        
        # Akım sensörleri
        for i in range(2):
            self.sensor_visibility["current"][i] = False
            self.current_checkboxes[i].setChecked(False)
            if hasattr(self, 'current_plots') and i < len(self.current_plots):
                plot_widget = self.current_plots[i].getViewWidget()
                plot_widget.setVisible(False)
        
        # EKG sensörü
        self.sensor_visibility["ecg"] = False
        self.ecg_checkbox.setChecked(False)
        if hasattr(self, 'ecg_plot'):
            plot_widget = self.ecg_plot.getViewWidget()
            plot_widget.setVisible(False)
    
    def toggle_plotting(self):
        """Grafik çizimini durdurur/başlatır"""
        self.plotting_active = not self.plotting_active
        
        if self.plotting_active:
            self.stop_plotting_btn.setText("Grafik Çizimini Durdur")
            self.stop_plotting_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
            self.update_status("Grafik çizimi başlatıldı")
        else:
            self.stop_plotting_btn.setText("Grafik Çizimini Başlat")
            self.stop_plotting_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            self.update_status("Grafik çizimi durduruldu")
    
    def update_status(self, message):
        """Durum çubuğunu güncelle"""
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message, 3000)
    
    def keyPressEvent(self, event):
        """Tuş basma olaylarını işle"""
        if event.key() == Qt.Key.Key_A:
            if hasattr(self, 'realtime_graph'):
                self.realtime_graph.enableAutoRange()
                self.update_status("Otomatik ölçeklendirme etkinleştirildi")
        super().keyPressEvent(event)
    
    def _perform_memory_cleanup(self):
        """Periyodik bellek temizliği işlemi"""
        current_time = time.time()
        
        if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
            self.last_memory_cleanup = current_time
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Bellek temizliği işlemi"""
        try:
            import gc
            gc.collect()
            
            # Bellek kullanımı kontrolü
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
        except ImportError:
            pass
        except Exception as e:
            self.update_status(f"Bellek temizliği hatası: {str(e)}")
    
    # Önbellek işlemleri kaldırıldı
    
    def on_sensor_data_updated(self):
        """Sensör verileri güncellendiğinde çağrılır"""
        if not self.auto_save_enabled:
            return
            
        current_time = time.time()
        # Belirli aralıklarla kaydet
        if current_time - self.last_save_time >= self.save_interval:
            self.save_sensor_data()
            self.last_save_time = current_time
            self.update_status(f"Sensör verileri otomatik olarak kaydedildi: {time.strftime('%H:%M:%S')}")
    
    def save_sensor_data_to_excel(self):
        """Sensör verilerini Excel dosyasına kaydet"""
        try:
            self.save_sensor_data()
            return True
        except Exception as e:
            self.update_status(f"Excel kaydetme hatası: {str(e)}")
            return False
    
    def closeEvent(self, event):
        """Pencere kapatıldığında kaynakları temizle"""
        try:
            self._running = False
            
            # Pencere kapatılırken sensör verilerini kaydet
            self.save_sensor_data_to_excel()
            self.update_status("Sensör verileri kapatılırken kaydedildi.")
            
            if hasattr(self, 'plot_timer'):
                self.plot_timer.stop()
            
            if hasattr(self, 'memory_timer'):
                self.memory_timer.stop()
            
            if hasattr(self, 'udp_socket') and self.udp_socket:
                self.udp_socket.close()
                
            if hasattr(self, 'udp_thread') and self.udp_thread.is_alive():
                self.udp_thread.join(1.0)
                
        except Exception as e:
            self.update_status(f"Kapatma hatası: {str(e)}")
        
        event.accept()

# Set PyQtGraph options
pg.setConfigOption('background', '#1e1e2e')

class SignalGeneratorWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("8-Channel PWM Signal Generator")
        self.setMinimumSize(1000, 800)
        
        # Initialize instance variables
        self.freq_spins = {}
        self.duty_spins = {}
        self.start_btns = {}
        self.stop_btns = {}
        
        # PWM durumunu takip etmek için değişkenler
        self.pwm_status = {ch: False for ch in range(1, 9)}  # Her kanal için PWM durumu (açık/kapalı)
        self.pwm_settings = {ch: {'freq': 1000, 'duty': 50} for ch in range(1, 9)}  # Her kanal için frekans ve duty cycle değerleri
        
        # Initialize UDP socket for sending commands
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Broadcast yerine doğrudan ESP32 IP'si kullanılacak
        esp32_ip = "192.168.244.194"  # ESP32'nin seri monitörde görünen IP'si
        self.udp_address = (esp32_ip, 4211)  # ESP32 IP adresi ve port
        
        # Initialize UI
        self.init_ui()
        
        # Update status
        self.update_status("Ready - Using direct ESP32 connection")
        
        # Otomatik komut gönderme işlemi iptal edildi
        # Pencere açıldığında, kaydedilen PWM ayarlarını sadece UI'a yükle (komut göndermeden)
        self.load_pwm_settings_ui_only()
                
    def save_pwm_settings(self):
        """PWM ayarlarını kalıcı olarak kaydet"""
        settings = QSettings("SignalGenerator", "PWMSettings")
        
        # PWM durumunu kaydet
        for ch in range(1, 9):
            settings.setValue(f"pwm_status_{ch}", self.pwm_status[ch])
            settings.setValue(f"pwm_freq_{ch}", self.pwm_settings[ch]['freq'])
            settings.setValue(f"pwm_duty_{ch}", self.pwm_settings[ch]['duty'])
        
        # Ayarları diske yaz
        settings.sync()
        self.update_status("PWM settings saved")
    
    def load_pwm_settings_ui_only(self):
        """Kaydedilen PWM ayarlarını sadece UI'a yükle (komut göndermeden)"""
        settings = QSettings("SignalGenerator", "PWMSettings")
        
        # Tüm kanallar için ayarları yükle
        for ch in range(1, 9):
            # Kaydedilen değerleri al, yoksa varsayılan değerleri kullan
            self.pwm_status[ch] = settings.value(f"pwm_status_{ch}", False, type=bool)
            self.pwm_settings[ch]['freq'] = settings.value(f"pwm_freq_{ch}", 1000, type=int)
            self.pwm_settings[ch]['duty'] = settings.value(f"pwm_duty_{ch}", 50, type=float)
            
            # Kaydedilen frekans ve duty cycle değerlerini ayarla
            freq = self.pwm_settings[ch]['freq']
            duty = self.pwm_settings[ch]['duty']
            
            # Kanal 1 için özel UI güncellemesi
            if ch == 1:
                # Sadece UI'ı güncelle, komut gönderme
                self.coil1_freq_spin.setValue(freq)
                self.coil1_duty_spin.setValue(int(duty))  # int'e dönüştürerek DeprecationWarning'i önle
                
                # Eğer kanal aktifse, sadece UI durumunu güncelle
                if self.pwm_status.get(1, False):
                    # UI durumunu güncelle
                    self.coil1_status.setText(f"Ready at {freq} Hz, {duty}% duty")
                    self.coil1_status.setStyleSheet("color: #00ff7f; font-weight: bold;")
                    self.coil1_start_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00ff7f;
                            color: #000000;
                            border: 2px solid #00ff7f;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                    """)
                    self.coil1_stop_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #2a2a3a;
                            color: #ff5500;
                            border: 2px solid #ff5500;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                    """)
            # Diğer kanallar için UI güncellemesi (2-8)
            else:
                # Kanal indeksi 0'dan başladığı için ch-1 kullanıyoruz
                channel_idx = ch - 1
                if channel_idx in self.freq_spins and channel_idx in self.duty_spins:
                    # Frekans ve duty değerlerini UI'a yansıt
                    self.freq_spins[channel_idx].setValue(freq)
                    self.duty_spins[channel_idx].setValue(duty)
                    
                    # Buton durumlarını güncelle
                    if channel_idx in self.start_btns and channel_idx in self.stop_btns:
                        if self.pwm_status.get(ch, False):
                            # PWM aktifse, start butonunu seç, stop butonunu seçme
                            self.start_btns[channel_idx].setChecked(True)
                            self.stop_btns[channel_idx].setChecked(False)
                            # Start butonunu yeşil yap
                            self.start_btns[channel_idx].setStyleSheet("""
                                QPushButton {
                                    background-color: #00ff7f;
                                    color: #000000;
                                    border: 2px solid #00ff7f;
                                    border-radius: 6px;
                                    padding: 8px;
                                    font-weight: bold;
                                    font-size: 14px;
                                }
                            """)
                            # Stop butonunu normal stile çevir
                            self.stop_btns[channel_idx].setStyleSheet("""
                                QPushButton {
                                    background-color: #2a2a3a;
                                    color: #ff5500;
                                    border: 2px solid #ff5500;
                                    border-radius: 6px;
                                    padding: 8px;
                                    font-weight: bold;
                                    font-size: 14px;
                                }
                            """)
                        else:
                            # PWM aktif değilse, stop butonunu seç, start butonunu seçme
                            self.start_btns[channel_idx].setChecked(False)
                            self.stop_btns[channel_idx].setChecked(True)
                            # Stop butonunu kırmızı yap
                            self.stop_btns[channel_idx].setStyleSheet("""
                                QPushButton {
                                    background-color: #ff5500;
                                    color: #000000;
                                    border: 2px solid #ff5500;
                                    border-radius: 6px;
                                    padding: 8px;
                                    font-weight: bold;
                                    font-size: 14px;
                                }
                            """)
                            # Start butonunu normal stile çevir
                            self.start_btns[channel_idx].setStyleSheet("""
                                QPushButton {
                                    background-color: #2a2a3a;
                                    color: #00ff7f;
                                    border: 2px solid #00ff7f;
                                    border-radius: 6px;
                                    padding: 8px;
                                    font-weight: bold;
                                    font-size: 14px;
                                }
                            """)
    
    def load_pwm_settings(self):
        """Kaydedilen PWM ayarlarını yükle ve aktif kanalları başlat"""
        settings = QSettings("SignalGenerator", "PWMSettings")
        
        # Tüm kanallar için ayarları yükle
        for ch in range(1, 9):
            # Kaydedilen değerleri al, yoksa varsayılan değerleri kullan
            self.pwm_status[ch] = settings.value(f"pwm_status_{ch}", False, type=bool)
            self.pwm_settings[ch]['freq'] = settings.value(f"pwm_freq_{ch}", 1000, type=int)
            self.pwm_settings[ch]['duty'] = settings.value(f"pwm_duty_{ch}", 50, type=float)
            
            # Kaydedilen frekans ve duty cycle değerlerini ayarla
            freq = self.pwm_settings[ch]['freq']
            duty = self.pwm_settings[ch]['duty']
            
            # Kanal 1 için özel UI güncellemesi
            if ch == 1:
                # UI'ı güncelle
                self.coil1_freq_spin.setValue(freq)
                self.coil1_duty_spin.setValue(duty)
                
                # Komutları gönder
                self.publish_command(1, "freq", freq)
                self.publish_command(1, "duty", duty)
                
                # Eğer kanal aktifse, başlat
                if self.pwm_status.get(1, False):
                    self.publish_command(1, "start")
                    
                    # UI durumunu güncelle
                    self.coil1_status.setText(f"Running at {freq} Hz, {duty}% duty")
                    self.coil1_status.setStyleSheet("color: #00ff7f; font-weight: bold;")
                    self.coil1_start_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00ff7f;
                            color: #000000;
                            border: 2px solid #00ff7f;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                    """)
                    self.coil1_stop_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #2a2a3a;
                            color: #ff5500;
                            border: 2px solid #ff5500;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                        QPushButton:hover {
                            background-color: #ff5500;
                            color: #000000;
                        }
                    """)
            
            # Tüm kanallar için genel UI güncellemesi
            if ch-1 in self.freq_spins and ch-1 in self.duty_spins:
                # Spin box değerlerini güncelle
                self.freq_spins[ch-1].setValue(freq)
                self.duty_spins[ch-1].setValue(duty)
                
                # Buton durumlarını güncelle
                if ch-1 in self.start_btns and ch-1 in self.stop_btns:
                    if self.pwm_status.get(ch, False):
                        # PWM aktifse, start butonunu seç, stop butonunu seçme
                        self.start_btns[ch-1].setChecked(True)
                        self.stop_btns[ch-1].setChecked(False)
                        # Start butonunu yeşil yap
                        self.start_btns[ch-1].setStyleSheet("""
                            QPushButton {
                                background-color: #00ff7f;
                                color: #000000;
                                border: 2px solid #00ff7f;
                                border-radius: 6px;
                                padding: 8px;
                                font-weight: bold;
                                font-size: 14px;
                            }
                        """)
                        # Stop butonunu normal stile çevir
                        self.stop_btns[ch-1].setStyleSheet("""
                            QPushButton {
                                background-color: #2a2a3a;
                                color: #ff5500;
                                border: 2px solid #ff5500;
                                border-radius: 6px;
                                padding: 8px;
                                font-weight: bold;
                                font-size: 14px;
                            }
                            QPushButton:hover {
                                background-color: #ff5500;
                                color: #000000;
                            }
                        """)
                    else:
                        # PWM aktif değilse, stop butonunu seç, start butonunu seçme
                        self.start_btns[ch-1].setChecked(False)
                        self.stop_btns[ch-1].setChecked(True)
                        # Stop butonunu kırmızı yap
                        self.stop_btns[ch-1].setStyleSheet("""
                            QPushButton {
                                background-color: #ff5500;
                                color: #000000;
                                border: 2px solid #ff5500;
                                border-radius: 6px;
                                padding: 8px;
                                font-weight: bold;
                                font-size: 14px;
                            }
                        """)
                        # Start butonunu normal stile çevir
                        self.start_btns[ch-1].setStyleSheet("""
                            QPushButton {
                                background-color: #2a2a3a;
                                color: #00ff7f;
                                border: 2px solid #00ff7f;
                                border-radius: 6px;
                                padding: 8px;
                                font-weight: bold;
                                font-size: 14px;
                            }
                            QPushButton:hover {
                                background-color: #00ff7f;
                                color: #000000;
                            }
                        """)
                
                # Komutları gönder (1-4 kanalları için)
                if 1 <= ch <= 4:  # Sadece 1-4 kanalları UDP ile destekleniyor
                    self.publish_command(ch, "freq", freq)
                    self.publish_command(ch, "duty", duty)
                    
                    # Eğer kanal aktifse, başlat
                    if self.pwm_status.get(ch, False):
                        self.publish_command(ch, "start")
        
        self.update_status("PWM settings loaded")
    
    def send_udp_command(self, channel, cmd_type, value=None):
        """
        Send a command via UDP to control PWM parameters
        
        Args:
            channel (int): Channel number (1-8)
            cmd_type (str): Command type ("start", "stop", "freq", or "duty")
            value: Value for frequency or duty cycle (ignored for start/stop)
        """
        if channel < 1 or channel > 8:  # Handle all 8 channels
            return False
            
        try:
            # Create command dictionary in ESP32 expected format
            cmd = {}
            
            # Add coil_id to the command so ESP32 can filter commands
            cmd["coil_id"] = channel
            
            if cmd_type == "start":
                cmd["pwm_start"] = True
            elif cmd_type == "stop":
                cmd["pwm_stop"] = True
            elif cmd_type == "freq" and value is not None:
                cmd["pwm_freq"] = int(value)
            elif cmd_type == "duty" and value is not None:
                cmd["pwm_duty"] = float(value)
            else:
                print(f"Unknown command type: {cmd_type}")
                return False
            
            # Her ESP32 için ayrı IP adresi tanımı
            # Kullanıcı her ESP32 için kendi IP adresini burada tanımlayabilir
            udp_addresses = {
                1: ("192.168.244.200", 4211),  # ESP32 #1 - coil_id 1
                2: ("192.168.161.111", 4211),  # ESP32 #2 - coil_id 2
                3: ("192.168.161.93", 4211),  # ESP32 #3 - coil_id 3
                4: ("192.168.161.94", 4211),  # ESP32 #4 - coil_id 4
                5: ("192.168.161.95", 4211),  # ESP32 #5 - coil_id 5
                6: ("192.168.161.96", 4211),  # ESP32 #6 - coil_id 6
                7: ("192.168.161.97", 4211),  # ESP32 #7 - coil_id 7
                8: ("192.168.161.98", 4211),  # ESP32 #8 - coil_id 8
            }
            
            # Get the target address for this channel/coil_id
            target_address = udp_addresses.get(channel)
            if not target_address:
                print(f"No UDP address defined for coil {channel}")
                return False
            
            # Her kanal için işlem
            print(f"[COIL-{channel}] Sending PWM command: {cmd_type.upper()} {value if value is not None else ''}")
            # Komut gönderildiğinde UI'da gösterim
            self.update_status(f"Coil {channel} (Active): {cmd_type.upper()} {value if value is not None else ''}")
            
            # Her kanal için start/stop durumunda UI güncellemesi
            if cmd_type == "start" and hasattr(self, 'start_btns') and len(self.start_btns) >= channel:
                # Start butonunu yeşil yap
                self.start_btns[channel-1].setStyleSheet("""
                    QPushButton {
                        background-color: #00ff7f;
                        color: #000000;
                        border: 2px solid #00ff7f;
                        border-radius: 6px;
                        padding: 8px;
                        font-weight: bold;
                        font-size: 14px;
                    }
                """)
                if hasattr(self, 'stop_btns') and len(self.stop_btns) >= channel:
                    # Stop butonunu normal stile çevir
                    self.stop_btns[channel-1].setStyleSheet("""
                        QPushButton {
                            background-color: #2a2a3a;
                            color: #ff5500;
                            border: 2px solid #ff5500;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                        QPushButton:hover {
                            background-color: #ff5500;
                            color: #000000;
                        }
                    """)
            elif cmd_type == "stop" and hasattr(self, 'stop_btns') and len(self.stop_btns) >= channel:
                # Stop butonunu kırmızı yap
                self.stop_btns[channel-1].setStyleSheet("""
                    QPushButton {
                        background-color: #ff5500;
                        color: #000000;
                        border: 2px solid #ff5500;
                        border-radius: 6px;
                        padding: 8px;
                        font-weight: bold;
                        font-size: 14px;
                    }
                """)
                if hasattr(self, 'start_btns') and len(self.start_btns) >= channel:
                    # Start butonunu normal stile çevir
                    self.start_btns[channel-1].setStyleSheet("""
                        QPushButton {
                            background-color: #2a2a3a;
                            color: #00ff7f;
                            border: 2px solid #00ff7f;
                            border-radius: 6px;
                            padding: 8px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                        QPushButton:hover {
                            background-color: #00ff7f;
                            color: #000000;
                        }
                    """)
            
            # Convert to JSON and send via UDP to specific ESP32
            message = json.dumps(cmd).encode('utf-8')
            self.udp_socket.sendto(message, target_address)
            
            # Debug: Print the exact message being sent
            print(f"[DEBUG] Coil {channel}: Sending UDP message: {message.decode('utf-8')}")
            print(f"[DEBUG] UDP Address: {target_address}")
            
            return True
            
        except Exception as e:
            print(f"UDP command failed: {type(e).__name__}")
            self.update_status(f"Command failed: {type(e).__name__}")
            return False
    
    def publish_command(self, channel, cmd_type, value=""):
        """
        Send UDP command for a specific channel
        
        Args:
            channel (int): Channel number (1-8)
            cmd_type (str): Command type ("start", "stop", "freq", or "duty")
            value: Value for frequency or duty cycle (ignored for start/stop)
                
        Returns:
            bool: True if sent successfully, False otherwise
        """
        # PWM ayarlarını güncelle
        if cmd_type == "freq" and value:
            self.pwm_settings[channel]['freq'] = value
        elif cmd_type == "duty" and value:
            self.pwm_settings[channel]['duty'] = value
        elif cmd_type == "start":
            self.pwm_status[channel] = True
        elif cmd_type == "stop":
            self.pwm_status[channel] = False
            
        # For all channels 1-8, use UDP
        result = False
        if 1 <= channel <= 8:
            result = self.send_udp_command(channel, cmd_type, value)
        else:
            self.update_status(f"Channel {channel} not supported")
            result = False
        
        # Ayarları kaydet (her değişiklikte)
        if result and cmd_type in ["freq", "duty", "start", "stop"]:
            self.save_pwm_settings()
            
        return result

    def start_coil1_pwm(self):
        """Start PWM for Coil ID 1"""
        # Get current frequency and duty cycle values
        freq = self.coil1_freq_spin.value()
        duty = self.coil1_duty_spin.value()
        
        # Update UI
        self.coil1_status.setText("Starting...")
        self.coil1_status.setStyleSheet("color: #ffff00; font-weight: bold;")
        
        # Send frequency and duty commands first
        if not self.publish_command(1, "freq", freq):
            self.coil1_status.setText("Error setting frequency")
            self.coil1_status.setStyleSheet("color: #ff0000; font-weight: bold;")
            return
            
        if not self.publish_command(1, "duty", duty):
            self.coil1_status.setText("Error setting duty cycle")
            self.coil1_status.setStyleSheet("color: #ff0000; font-weight: bold;")
            return
        
        # Send start command
        if self.publish_command(1, "start"):
            # publish_command zaten pwm_status ve pwm_settings'i güncelliyor, tekrar güncellemeye gerek yok
            
            self.coil1_status.setText(f"Running at {freq} Hz, {duty}% duty")
            self.coil1_status.setStyleSheet("color: #00ff7f; font-weight: bold;")
            self.coil1_start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00ff7f;
                    color: #000000;
                    border: 2px solid #00ff7f;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 14px;
                }
            """)
            self.coil1_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2a3a;
                    color: #ff5500;
                    border: 2px solid #ff5500;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #ff5500;
                    color: #000000;
                }
            """)
        else:
            self.coil1_status.setText("Error starting PWM")
            self.coil1_status.setStyleSheet("color: #ff0000; font-weight: bold;")
    
    def stop_coil1_pwm(self):
        """Stop PWM for Coil ID 1"""
        # Update UI
        self.coil1_status.setText("Stopping...")
        self.coil1_status.setStyleSheet("color: #ffff00; font-weight: bold;")
        
        # Send stop command
        if self.publish_command(1, "stop"):
            # PWM durumunu publish_command zaten güncelliyor, ek güncelleme gerekmez
            
            self.coil1_status.setText("Stopped")
            self.coil1_status.setStyleSheet("color: #ff5500; font-weight: bold;")
            self.coil1_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff5500;
                    color: #000000;
                    border: 2px solid #ff5500;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 14px;
                }
            """)
            self.coil1_start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2a3a;
                    color: #00ff7f;
                    border: 2px solid #00ff7f;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #00ff7f;
                    color: #000000;
                }
            """)
        else:
            self.coil1_status.setText("Error stopping PWM")
            self.coil1_status.setStyleSheet("color: #ff0000; font-weight: bold;")
    
    def update_coil1_freq(self, freq):
        """Update frequency for Coil ID 1"""
        self.send_udp_command(1, "freq", freq)
        self.coil1_status.setText(f"Frequency set to {freq} Hz")
    
    def update_coil1_duty(self, duty):
        """Update duty cycle for Coil ID 1"""
        self.send_udp_command(1, "duty", duty)
        self.coil1_status.setText(f"Duty cycle set to {duty}%")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # PWM ayarlarını kaydet
        self.save_pwm_settings()
        
        # Close UDP socket if it exists
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()
            
        # Close UDP PWM socket if it exists
        if hasattr(self, 'udp_pwm_socket'):
            self.udp_pwm_socket.close()
            
        # Call parent close event
        super().closeEvent(event)

    def init_ui(self):
        # Initialize status bar first
        self.update_status("Initializing...")
        
        # Coil ID 1 için özel başlık ekle
        self.setWindowTitle("8-Channel PWM Signal Generator")
        
        # Ana düzen oluştur
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Coil ID 1 için özel kontrol paneli
        coil1_group = QGroupBox("Coil ID 1 - ESP32 Control Panel")
        coil1_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a2a;
                border: 2px solid #00ff7f;
                border-radius: 8px;
                margin-top: 1ex;
                font-weight: bold;
                color: #00ff7f;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #1a1a2a;
            }
        """)
        
        coil1_layout = QGridLayout(coil1_group)
        
        # ESP32 veri hızı göstergesi
        data_rate_label = QLabel("ESP32 Data Rate:")
        data_rate_label.setStyleSheet("color: #00ff7f; font-weight: bold;")
        self.data_rate_label = QLabel("Waiting...")
        self.data_rate_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        coil1_layout.addWidget(data_rate_label, 0, 0)
        coil1_layout.addWidget(self.data_rate_label, 0, 1, 1, 2)
        
        # Frekans kontrolü
        freq_label = QLabel("Frequency (Hz):")
        freq_label.setStyleSheet("color: #00ff7f; font-weight: bold;")
        self.coil1_freq_spin = QSpinBox()
        self.coil1_freq_spin.setRange(1, 20000)
        self.coil1_freq_spin.setValue(100)
        self.coil1_freq_spin.setSingleStep(10)
        self.coil1_freq_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #00ff7f;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
        """)
        
        # Duty cycle kontrolü
        duty_label = QLabel("Duty Cycle (%):")
        duty_label.setStyleSheet("color: #00ff7f; font-weight: bold;")
        self.coil1_duty_spin = QSpinBox()
        self.coil1_duty_spin.setRange(0, 100)
        self.coil1_duty_spin.setValue(50)
        self.coil1_duty_spin.setSingleStep(5)
        self.coil1_duty_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #00ff7f;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
        """)
        
        # Start/Stop butonları
        self.coil1_start_btn = QPushButton("Start PWM")
        self.coil1_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3a;
                color: #00ff7f;
                border: 2px solid #00ff7f;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00ff7f;
                color: #000000;
            }
        """)
        
        self.coil1_stop_btn = QPushButton("Stop PWM")
        self.coil1_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3a;
                color: #ff5500;
                border: 2px solid #ff5500;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff5500;
                color: #000000;
            }
        """)
        
        # Durum göstergesi
        status_label = QLabel("Status:")
        status_label.setStyleSheet("color: #00ff7f; font-weight: bold;")
        self.coil1_status = QLabel("Ready")
        self.coil1_status.setStyleSheet("color: #ffffff; font-weight: bold;")
        
        # Kontrolleri düzene ekle
        coil1_layout.addWidget(freq_label, 1, 0)
        coil1_layout.addWidget(self.coil1_freq_spin, 1, 1)
        coil1_layout.addWidget(duty_label, 2, 0)
        coil1_layout.addWidget(self.coil1_duty_spin, 2, 1)
        coil1_layout.addWidget(self.coil1_start_btn, 1, 2)
        coil1_layout.addWidget(self.coil1_stop_btn, 2, 2)
        coil1_layout.addWidget(status_label, 3, 0)
        coil1_layout.addWidget(self.coil1_status, 3, 1, 1, 2)
        
        # Buton bağlantıları
        self.coil1_start_btn.clicked.connect(lambda: self.start_coil1_pwm())
        self.coil1_stop_btn.clicked.connect(lambda: self.stop_coil1_pwm())
        self.coil1_freq_spin.valueChanged.connect(lambda val: self.update_coil1_freq(val))
        self.coil1_duty_spin.valueChanged.connect(lambda val: self.update_coil1_duty(val))
        
        # Ana düzene ekle
        main_layout.addWidget(coil1_group)
        
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #3d206b;
                color: #ffffff;
                border: 1px solid #e3e8f0;
                border-radius: 4px;
                padding: 6px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3d206b;
                color: #ffffff;
                border-radius: 8px;
                padding: 6px 16px;
                min-height: 40px;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #4a2a7a;
            }
            QPushButton#startBtn:checked {
                background-color: #2e7d32;
            }
            QPushButton#stopBtn:checked {
                background-color: #c62828;
            }
            QGroupBox {
                border: 1px solid #e3e8f0;
                border-radius: 8px;
                padding: 10px;
                margin-top: 6px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #ffffff;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #3d206b;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #7e57c2;
                min-height: 20px;
                border-radius: 5px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        grid = QGridLayout()
        grid.setSpacing(15)
        
        # Create channel controls
        for i in range(8):
            row = i // 2
            col = i % 2
            
            group = QGroupBox(f"Channel {i+1}")
            group.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #3a3a4a;
                    border-radius: 5px;
                    margin-top: 1em;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)
            
            channel_layout = QVBoxLayout(group)
            channel_layout.setContentsMargins(10, 15, 10, 10)
            channel_layout.setSpacing(10)
            
            # Frequency control
            freq_layout = QHBoxLayout()
            freq_label = QLabel("Freq (Hz):")
            freq_label.setStyleSheet("color: #e0e0e0;")
            
            self.freq_spins[i] = QDoubleSpinBox()
            self.freq_spins[i].setRange(1, 100000)
            self.freq_spins[i].setValue(1000)
            self.freq_spins[i].setDecimals(0)
            self.freq_spins[i].setSingleStep(100)
            self.freq_spins[i].setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #2a2a3a;
                    color: #ffffff;
                    border: 1px solid #3a3a4a;
                    border-radius: 3px;
                    padding: 3px;
                }
            """)
            self.freq_spins[i].valueChanged.connect(lambda value, ch=i+1: self.update_frequency(ch, value))
            
            freq_layout.addWidget(freq_label)
            freq_layout.addWidget(self.freq_spins[i])
            
            # Duty cycle control
            duty_layout = QHBoxLayout()
            duty_label = QLabel("Duty (%):")
            duty_label.setStyleSheet("color: #e0e0e0;")
            
            self.duty_spins[i] = QDoubleSpinBox()
            self.duty_spins[i].setRange(0.1, 99.9)
            self.duty_spins[i].setValue(50.0)
            self.duty_spins[i].setDecimals(1)
            self.duty_spins[i].setSingleStep(0.1)
            self.duty_spins[i].setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #2a2a3a;
                    color: #ffffff;
                    border: 1px solid #3a3a4a;
                    border-radius: 3px;
                    padding: 3px;
                }
            """)
            self.duty_spins[i].valueChanged.connect(lambda value, ch=i+1: self.update_duty(ch, value))
            
            duty_layout.addWidget(duty_label)
            duty_layout.addWidget(self.duty_spins[i])
            
            # Start/Stop buttons
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(5)
            
            self.start_btns[i] = QPushButton("Start")
            self.start_btns[i].setCheckable(True)
            self.start_btns[i].setStyleSheet("""
                QPushButton {
                    background-color: #2e7d32;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:checked {
                    background-color: #1b5e20;
                }
            """)
            self.start_btns[i].clicked.connect(lambda checked, ch=i+1: self.start_pwm(ch, checked))
            
            self.stop_btns[i] = QPushButton("Stop")
            self.stop_btns[i].setCheckable(True)
            self.stop_btns[i].setChecked(True)
            self.stop_btns[i].setStyleSheet("""
                QPushButton {
                    background-color: #c62828;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:checked {
                    background-color: #8e0000;
                }
            """)
            self.stop_btns[i].clicked.connect(lambda checked, ch=i+1: self.stop_pwm(ch, checked))
            
            btn_layout.addWidget(self.start_btns[i])
            btn_layout.addWidget(self.stop_btns[i])
            
            # Add all to channel layout
            channel_layout.addLayout(freq_layout)
            channel_layout.addLayout(duty_layout)
            channel_layout.addLayout(btn_layout)
            
            # Add group to grid
            grid.addWidget(group, row, col)

        # Create a widget to contain the grid
        container = QWidget()
        container.setLayout(grid)
        
        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #2a2a3a;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #7e57c2;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        main_layout.addWidget(scroll, 1)  # Add scroll area with stretch factor

        # Global controls - Start All ve Stop All butonları
        global_controls = QHBoxLayout()
        global_controls.setSpacing(15)
        global_controls.setContentsMargins(10, 10, 10, 10)
        
        # Start All butonu
        self.start_all_btn = QPushButton("START ALL")
        self.start_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1b5e20;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)
        self.start_all_btn.clicked.connect(self.start_all_pwm)
        
        # Stop All butonu
        self.stop_all_btn = QPushButton("STOP ALL")
        self.stop_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #8e0000;
            }
            QPushButton:pressed {
                background-color: #d32f2f;
            }
        """)
        self.stop_all_btn.clicked.connect(self.stop_all_pwm)
        
        global_controls.addWidget(self.start_all_btn)
        global_controls.addWidget(self.stop_all_btn)
        
        main_layout.addLayout(global_controls)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status()
        
    def update_frequency(self, channel, freq):
        """Update PWM frequency for a channel"""
        self.publish_command(channel, "freq", int(freq))
        self.status_bar.showMessage(f"Channel {channel}: Frequency → {int(freq)} Hz")
        # Ayarları kalıcı olarak kaydet
        self.save_pwm_settings()
    
    def update_duty(self, channel, duty):
        """Update PWM duty cycle for a channel"""
        self.publish_command(channel, "duty", duty)
        self.status_bar.showMessage(f"Channel {channel}: Duty → {duty:.1f}%")
        # Ayarları kalıcı olarak kaydet
        self.save_pwm_settings()
    
    def start_pwm(self, channel, checked):
        # Update frequency and duty first
        freq = self.freq_spins[channel-1].value()
        duty = self.duty_spins[channel-1].value()
        
        # Ayarları kaydet
        self.pwm_settings[channel]['freq'] = freq
        self.pwm_settings[channel]['duty'] = duty
        
        # Publish frequency and duty updates
        if not self.publish_command(channel, "freq", freq):
            return
            
        if not self.publish_command(channel, "duty", duty):
            return
        
        # Uncheck stop button and start PWM after a small delay
        self.stop_btns[channel-1].setChecked(False)
        
        # Use QTimer for non-blocking delay
        QTimer.singleShot(10, lambda: self._delayed_start_pwm(channel))
        
    def _delayed_start_pwm(self, channel):
        """Helper method to start PWM after a small delay"""
        if self.publish_command(channel, "start"):
            # PWM durumunu güncelle
            self.pwm_status[channel] = True
            self.update_status(f"Channel {channel}: PWM started")
            # Ayarları kalıcı olarak kaydet
            self.save_pwm_settings()
    
    def stop_pwm(self, channel, checked):
        """Stop PWM generation for a channel"""
        if checked:
            self.start_btns[channel-1].setChecked(False)
            # Use QTimer for non-blocking delay
            QTimer.singleShot(10, lambda: self._delayed_stop_pwm(channel))
            
    def _delayed_stop_pwm(self, channel):
        """Helper method to handle PWM stop after delay"""
        if self.publish_command(channel, "stop"):
            # PWM durumunu güncelle ama ayarları sakla
            self.pwm_status[channel] = False
            self.update_status(f"Channel {channel}: PWM stopped")
            # Ayarları kalıcı olarak kaydet
            self.save_pwm_settings()
    
    def start_all_pwm(self):
        """Start all PWM channels simultaneously"""
        # Önce tüm kanalların frekans ve duty değerlerini ayarla
        for channel in range(1, 9):
            freq = self.freq_spins[channel-1].value()
            duty = self.duty_spins[channel-1].value()
            
            # Ayarları kaydet
            self.pwm_settings[channel]['freq'] = freq
            self.pwm_settings[channel]['duty'] = duty
            
            # Frekans ve duty değerlerini gönder
            self.publish_command(channel, "freq", freq)
            self.publish_command(channel, "duty", duty)
            
            # UI butonlarını güncelle
            self.start_btns[channel-1].setChecked(True)
            self.stop_btns[channel-1].setChecked(False)
        
        # Tüm kanalları aynı anda başlat (milisaniye hassasiyetinde)
        for channel in range(1, 9):
            # Start komutunu gönder
            self.publish_command(channel, "start")
            # PWM durumunu güncelle
            self.pwm_status[channel] = True
        
        # Ayarları kaydet
        self.save_pwm_settings()
        self.update_status("All PWM channels started simultaneously")
    
    def stop_all_pwm(self):
        """Stop all PWM channels simultaneously"""
        # Tüm kanalları aynı anda durdur
        for channel in range(1, 9):
            # UI butonlarını güncelle
            self.stop_btns[channel-1].setChecked(True)
            self.start_btns[channel-1].setChecked(False)
            
            # Stop komutunu gönder
            self.publish_command(channel, "stop")
            # PWM durumunu güncelle
            self.pwm_status[channel] = False
        
        # Ayarları kaydet
        self.save_pwm_settings()
        self.update_status("All PWM channels stopped simultaneously")
    
    def update_status(self, message=""):
        """Update connection status in status bar"""
        if not hasattr(self, 'status_bar'):
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(message if message else "Ready")

        self.resize(900, 700)




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # No serial port initialization in main window
        # Each child window will handle its own serial connection

    def read_serial_data(self):
        """Read data from serial port in a separate thread with STX/ETX framing"""
        buffer = bytearray()
        while hasattr(self, 'serial_running') and self.serial_running and hasattr(self, 'serial_port') and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    # Read all available bytes
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer.extend(data)
                    
                    # Process all complete messages in buffer (STX ... ETX)
                    while True:
                        # Find STX (start of message)
                        stx_pos = buffer.find(0x02)  # STX byte
                        if stx_pos == -1:  # No STX found
                            buffer = bytearray()  # Discard data before STX
                            break
                            
                        # Find ETX after STX
                        etx_pos = buffer.find(0x03, stx_pos + 1)  # ETX after STX
                        if etx_pos == -1:  # No complete message yet
                            # Keep the buffer from STX onwards for next read
                            buffer = buffer[stx_pos:]
                            break
                            
                        # Extract message (excluding STX and ETX)
                        message = buffer[stx_pos+1:etx_pos].decode('ascii', errors='ignore')
                        # Remove the processed message from buffer
                        buffer = buffer[etx_pos+1:]
                        
                        # Process the complete message
                        if message.strip():
                            self.process_serial_line(message.strip())
                else:
                    time.sleep(0.01)  # Small delay to prevent high CPU usage
            except Exception as e:
                print(f"[ERROR] Serial read error: {e}")
                buffer = bytearray()  # Reset buffer on error
                time.sleep(0.1)  # Wait a bit before retrying

    def process_serial_line(self, line):
        """Process a line of text received from the serial port"""
        try:
            print(f"[RECV] {line}")
            
            # Update status based on received messages
            if "PWM: ON" in line:
                if hasattr(self, 'pwm_status'):
                    self.pwm_status.setText("Status: ON")
                    self.pwm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            elif "PWM: OFF" in line:
                if hasattr(self, 'pwm_status'):
                    self.pwm_status.setText("Status: OFF")
                    self.pwm_status.setStyleSheet("color: #f44336; font-weight: bold;")
            elif "ERROR" in line:
                if hasattr(self, 'statusBar'):
                    self.statusBar.showMessage(f"Error: {line}", 5000)
        except Exception as e:
            print(f"[ERROR] Error processing serial line: {e}")

    def closeEvent(self, event):
        """Handle application close"""
        # Clean up serial port
        if hasattr(self, 'serial_port') and self.serial_port and self.serial_port.is_open:
            # Note: We're not stopping PWM here - it will keep running
            self.serial_port.close()
        event.accept()
    def __init__(self):
        super().__init__()
        # --- Working Time Persistence (move to top of __init__) ---
        self.working_time_file = 'working_time.txt'
        self.working_seconds = 0
        if os.path.exists(self.working_time_file):
            try:
                with open(self.working_time_file, 'r') as f:
                    self.working_seconds = int(f.read().strip())
            except Exception:
                self.working_seconds = 0

        self.setWindowTitle("PEMF Medical System")
        self.setGeometry(100, 100, 1540, 900)
        self.setMinimumSize(1280, 800)
        
        # Gradient background for central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
        """)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Top Bar ---
        top_bar_widget = QWidget()
        top_bar_widget.setStyleSheet("""
            background: transparent;
            border-radius: 18px;
            margin: 8px 32px 0 32px;  /* Top:8px, sides:32px */
            padding: 0;
        """)
        top_bar_layout = QHBoxLayout(top_bar_widget)
        top_bar_layout.setContentsMargins(32, 6, 32, 6)  # biraz artırdım top/bottom padding
        top_bar_layout.setSpacing(20)
        
        # Logo + Text
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(12)
        
        icon_label = QLabel()
        icon_pixmap = QPixmap("pemf_heart_emf_icon.png").scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(icon_pixmap)
        
        text_label = QLabel(
            "<b style='color:#fff; font-size:28px;'>PEMF Medical System</b> "
            "<span style='color:#6cffb0; font-size:16px;'>✔ Certified</span>"
        )
        text_label.setStyleSheet("color: #fff;")
        
        logo_layout.addWidget(icon_label)
        logo_layout.addWidget(text_label)
        
        logo_container = QWidget()
        logo_container.setLayout(logo_layout)
        
        top_bar_layout.addWidget(logo_container)
        top_bar_layout.addStretch(1)
        
        # Connected status label
        status_connected = QLabel("<span style='color:#6cffb0; font-size:15px;'>● Connected</span>")
        status_connected.setStyleSheet("background: transparent; border-radius: 0;")
        top_bar_layout.addWidget(status_connected)
        
        # Clock label
        self.clock = QLabel()
        self.clock.setStyleSheet("""
            color: #fff;
            font-size: 15px;
            margin-left: 24px;
            background: transparent;
            border-radius: 0;
        """)
        top_bar_layout.addWidget(self.clock)
        
        # Emergency stop button
        emergency_btn = QPushButton("EMERGENCY STOP")
        emergency_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff5e62, stop:1 #ff9966);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 32px;
            margin-left: 32px;
        """)
        top_bar_layout.addWidget(emergency_btn)
        
        main_layout.addWidget(top_bar_widget)
        
        # --- Timer for updating live clock ---
        timer = QTimer(self)
        timer.timeout.connect(self.update_clock)
        timer.start(1000)
        self.update_clock()
        
        # --- Main Content Area ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(32, 16, 32, 16)
        content_layout.setSpacing(32)
        main_layout.addLayout(content_layout, stretch=1)
        # Sidebar
        sidebar = QVBoxLayout()
        sidebar.setSpacing(18)
        sidebar.setContentsMargins(0, 0, 0, 0)
        
        sidebar_widget = QWidget()
        sidebar_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sidebar_widget.setMinimumWidth(280)
        sidebar_widget.setMaximumWidth(500)
        sidebar_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
            padding: 16px;
            border-radius: 16px;
        """)
        sidebar_widget.setLayout(sidebar)
        content_layout.addWidget(sidebar_widget, 1)  # Stretch 1 ile esneklik verildi
        
        # Sidebar başlığı
        sidebar_title = QLabel("Sistem Parametreleri")
        sidebar_title.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold; margin: 0;")
        sidebar.addWidget(sidebar_title)
        
        # Scroll destekli giriş + bilgi paneli
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                width: 0px;
                height: 0px;
            }
        """)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(18)
        
        # 1. Giriş alanları
        param_labels = [
            "Port Adı Giriş", "Baudrate Giriş", "EKG Giriş", "Mesafe Giriş",
            "Sarım Sayısı Giriş", "Bobin Durumu Giriş", "Bobin Çeşidi Giriş"
        ]
        self.input_fields = []
        
        for label in param_labels:
            vbox = QVBoxLayout()
            vbox.setSpacing(6)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #fff; font-size: 14px; font-weight: bold; margin-left: 2px;")
            lbl.setFixedHeight(20)
        
            field = QLineEdit()
            field.setPlaceholderText("Enter value...")
            field.setStyleSheet(
                "background: #3d206b; color: white; border: none; border-radius: 10px; "
                "padding: 7px 12px; font-size: 14px;"
            )
            field.setFixedHeight(28)
        
            vbox.addWidget(lbl)
            vbox.addWidget(field)
            scroll_layout.addLayout(vbox)
            self.input_fields.append(field)
        
        
        # Bottom navigation bar
        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(32, 0, 32, 24)
        nav_bar.setSpacing(40)
        nav_widget = QWidget()
        nav_widget.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f); border-radius: 18px;")
        nav_widget.setLayout(nav_bar)
        nav_items = [
            ("\U0001F5A5", "AI Modelleri", self.open_AI_window),
            ("\U0001F4C8", "Sensör Verisi", self.open_sensor_data_window),
            ("\U0001F4BB", "Sinyal Üreteci", self.open_signal_generator),
            ("\U0001F5FA", "Dijital İkiz", self.open_digital_twin_window),
            ("\U0001F4CA", "KPI Paneli", self.open_kpi_dashboard),
            ("\U00002699", "Akıllı Tedavi", self.open_autonomous_mode)
        ]
        for icon, text, slot in nav_items:
            btn = QPushButton(f"{icon}  {text}")
            btn.setStyleSheet("background: transparent; color: #fff; border: none; font-size: 18px; font-weight: bold; padding: 18px 24px;")
            btn.clicked.connect(slot)
            nav_bar.addWidget(btn)
        main_layout.addWidget(nav_widget)
        
        
        # 2. Smart Treatment Card
        smart_treatment_card = QWidget()
        smart_treatment_card.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d206b, stop:1 #6c2b8f);
            border-radius: 12px;
            padding: 18px 16px;
        """)
        smart_treatment_layout = QVBoxLayout(smart_treatment_card)
        smart_treatment_layout.setSpacing(10)
        
        st_title = QLabel("\u23AF Aktif Tedavi")
        st_title.setStyleSheet("color: #6cffb0; font-size: 17px; font-weight: bold;")
        smart_treatment_layout.addWidget(st_title)
        
        def make_row(label_text, value_text, value_color="#fff", value_size="15px"):
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet("color: #fff; font-size: 15px; font-weight: bold;")
            value = QLabel(f"<span style='color:{value_color}; font-weight:bold; font-size:{value_size};'>{value_text}</span>")
            value.setStyleSheet("font-size: 15px; font-weight: bold;")
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(value)
            return row
        
        smart_treatment_layout.addLayout(make_row("Tedavi Türü:", "Ağrı Azaltma"))
        smart_treatment_layout.addLayout(make_row("Frekans:", "100 Hz", "#4f8cff"))
        smart_treatment_layout.addLayout(make_row("Yoğunluk:", "5 mT", "#4f8cff", "18px"))
        
        st_time_row = QHBoxLayout()
        st_time_icon = QLabel("\u23F1")
        st_time_icon.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold;")
        st_time_label = QLabel("Süre:")
        st_time_label.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold;")
        st_time_value = QLabel("30/30 dk")
        st_time_value.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold;")
        st_time_row.addWidget(st_time_icon)
        st_time_row.addWidget(st_time_label)
        st_time_row.addStretch(1)
        st_time_row.addWidget(st_time_value)
        smart_treatment_layout.addLayout(st_time_row)
        
        st_progress = QWidget()
        st_progress.setFixedHeight(14)
        st_progress.setStyleSheet("background: #4f8cff; border-radius: 7px;")
        smart_treatment_layout.addWidget(st_progress)
        
        st_status_row = QHBoxLayout()
        st_status = QLabel("<span style='color:#22c55e; font-size: 16px; font-weight: bold;'>● Çalışıyor</span>")
        st_status.setStyleSheet("background: #d1fae5; border-radius: 6px; padding: 4px 16px; margin-top: 6px; font-size: 16px;")
        st_status_row.addWidget(st_status)
        st_status_row.addStretch(1)
        smart_treatment_layout.addLayout(st_status_row)
        
        scroll_layout.addWidget(smart_treatment_card)
        
        # 3. KPI Card
        kpi_card = QWidget()
        kpi_card.setStyleSheet("background: transparent;")
        kpi_layout = QVBoxLayout(kpi_card)
        kpi_layout.setContentsMargins(0, 0, 0, 0)
        kpi_layout.setSpacing(8)
        
        kpi_title = QLabel("KPI Özeti")
        kpi_title.setStyleSheet("color: #fff; font-size: 15px; font-weight: bold;")
        kpi_layout.addWidget(kpi_title)
        
        def make_kpi(bg, icon, icon_color, label, label_color, value, value_color):
            widget = QWidget()
            widget.setStyleSheet(f"background: {bg}; border-radius: 8px; padding: 6px 10px;")
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(2)
            icon_lbl = QLabel(f"<span style='color:{icon_color}; font-size: 18px;'>{icon}</span>")
            text_lbl = QLabel(label)
            text_lbl.setStyleSheet(f"color: {label_color}; font-size: 13px; font-weight: bold;")
            val_lbl = QLabel(f"<span style='color:{value_color}; font-size: 16px; font-weight: bold;'>{value}</span>")
            layout.addWidget(icon_lbl)
            layout.addWidget(text_lbl)
            layout.addStretch(1)
            layout.addWidget(val_lbl)
            return widget
        
        kpi_layout.addWidget(make_kpi("#d1fae5", "⏫", "#22c55e", "Tedavi Etkinlik Oranı", "#166534", "78%", "#22c55e"))
        kpi_layout.addWidget(make_kpi("#fef9c3", "⚡", "#eab308", "Enerji Tüketimi", "#a16207", "0.12 kWh/tedavi", "#eab308"))
        kpi_layout.addWidget(make_kpi("#dbeafe", "⚙️", "#2563eb", "Cihaz Çalışma Oranı", "#2563eb", "95%", "#2563eb"))
        
        scroll_layout.addWidget(kpi_card)
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_content)
        sidebar.addWidget(scroll_area)
        
        # Center: Coil control panel
        center_panel = QWidget()
        center_panel.setStyleSheet("""
            background: rgba(40,20,80,0.85);
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        """)
        center_panel_layout = QVBoxLayout(center_panel)
        center_panel_layout.setContentsMargins(32, 32, 32, 32)
        center_panel_layout.setSpacing(24)
        
        # --- Sistem Durumu Başlık ---
        system_status_title = QLabel("<b style='color:#fff;font-size:22px;'>Sistem Durumu</b>")
        system_status_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        center_panel_layout.addWidget(system_status_title)
        
        # --- Durum Bilgileri Satırı ---
        status_row = QHBoxLayout()
        status_row.setSpacing(32)
        status_row.setContentsMargins(0, 0, 0, 0)
        
        def make_status_label(label, value, color):
            return QLabel(f"<span style='color:#bdb8e3;font-size:13px;'>{label}</span> "
                          f"<span style='color:{color}; font-size:14px; font-weight:bold;'>{value}</span>")
        
        status_row.addWidget(make_status_label("Durum:", "✔ Aktif", "#6cffb0"))
        status_row.addWidget(make_status_label("Bağlantı:", "📶 Bağlı", "#4f8cff"))
        status_row.addWidget(make_status_label("Sinyaller:", "⏸ 8/8", "#ffd86b"))
        
        status_row_widget = QWidget()
        status_row_widget.setLayout(status_row)
        status_row_widget.setStyleSheet("background: transparent;")
        center_panel_layout.addWidget(status_row_widget)
        
        # --- Çoklu Bobin Gerçek Zamanlı Grafik ---
        self.active_coils = set()  # Track which coils are active
        self.graph_start_time = None
        self.graph_time_data = []
        
        # Initialize data storage for each coil (1-8)
        self.graph_freq_data = {i: [] for i in range(1, 9)}
        self.graph_intensity_data = {i: [] for i in range(1, 9)}
        
        # Create the plot widget
        self.realtime_graph = pg.PlotWidget()
        self.realtime_graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.realtime_graph.setMinimumHeight(200)
        self.realtime_graph.setMaximumHeight(400)
        self.realtime_graph.setBackground(pg.mkColor(40, 20, 80, 200))
        self.realtime_graph.showGrid(x=True, y=True, alpha=0.3)
        self.realtime_graph.setLabel('left', 'Frekans (Hz) / Yoğunluk (mT)', color='#fff', size='14pt')
        self.realtime_graph.setLabel('bottom', 'Zaman (s)', color='#fff', size='12pt')
        self.realtime_graph.addLegend(offset=(10,10))
        
        # Create plot curves for each coil (frequency and intensity)
        self.freq_curves = {}
        self.intensity_curves = {}
        
        # Define colors for each coil
        colors = [
            '#FF5252', '#FF4081', '#E040FB', '#7C4DFF',
            '#536DFE', '#448AFF', '#40C4FF', '#18FFFF'
        ]
        
        # Create frequency and intensity curves for each coil
        for coil in range(1, 9):
            color = colors[coil-1]
            # Solid line for frequency
            self.freq_curves[coil] = self.realtime_graph.plot(
                pen=pg.mkPen(color=color, width=2), 
                name=f'Bobin {coil} Frekans',
                antialias=True
            )
            # Dashed line for intensity
            self.intensity_curves[coil] = self.realtime_graph.plot(
                pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine),
                name=f'Bobin {coil} Yoğunluk',
                antialias=True
            )
        
        center_panel_layout.addWidget(self.realtime_graph)
        
        # --- Bobin Kontrol Paneli Başlık ---
        coil_panel = QVBoxLayout()
        coil_panel.setSpacing(18)
        coil_panel_title = QLabel("<b style='color:#fff;font-size:22px;'>Bobin Kontrol Paneli</b>")
        coil_panel.addWidget(coil_panel_title)
        
        # --- 8 Adet Bobin Butonu ---
        grid = QGridLayout()
        grid.setSpacing(18)
        self.coil_buttons = []
        
        # Create coil buttons first
        for i in range(8):
            btn = QPushButton(f"⚡ Bobin-{i+1}")
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1ed6b5, stop:1 #3ed6b5);
                    color: white;
                    border: none;
                    border-radius: 14px;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 18px 0;
                }
                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff6b6b, stop:1 #ff8e8e);
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3ed6b5, stop:1 #5ed6b5);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00b894, stop:1 #00cec9);
                }
            """)
            btn.setCheckable(True)
            self.coil_buttons.append(btn)
            grid.addWidget(btn, i // 4, i % 4)  # 2 rows, 4 columns
            
            # Connect button click to toggle functionality
            btn.clicked.connect(lambda checked, ch=i+1: self.toggle_coil(ch))
            
        # Initially hide all curves
        for coil in range(1, 9):
            self.freq_curves[coil].hide()
            self.intensity_curves[coil].hide()   
        coil_panel.addLayout(grid)
        
        # --- Bobinleri Durdur Butonu ---
        stop_btn = QPushButton("🛑 Bobin Kapat!")
        stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff5e62, stop:1 #ff9966);
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 20px;
                font-weight: bold;
                padding: 18px 0;
                margin-top: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e45357, stop:1 #ff7f50);
            }
        """)
        stop_btn.setMinimumHeight(80)
        stop_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        stop_btn.clicked.connect(self.input_button_clicked)
        coil_panel.addWidget(stop_btn)
        
        center_panel_layout.addLayout(coil_panel)
        content_layout.addWidget(center_panel, stretch=2)
        
        # Right: Info panel
        info_panel = QWidget()
        info_panel.setStyleSheet("""
            background: rgba(40,20,80,0.85);
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        """)
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(22, 22, 22, 22)
        info_layout.setSpacing(12)
        
        info_title = QLabel("<b style='color:#fff;font-size:22px;'>Sistem Bilgileri</b>")
        info_layout.addWidget(info_title)
        
        # System Info Compact Card (vertical list, compact)
        system_info_card = QWidget()
        system_info_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        system_info_card.setMinimumWidth(250)
        system_info_card.setMaximumWidth(400)
        system_info_card.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d206b, stop:1 #6c2b8f);
            border-radius: 16px;
            padding: 16px;
            margin-top: 12px;
        """)
        
        system_info_layout = QVBoxLayout(system_info_card)
        system_info_layout.setContentsMargins(0, 0, 0, 0)
        system_info_layout.setSpacing(4)
        
        # Yardımcı fonksiyon
        def add_info_row(label_text, value_label, value_style=None):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
        
            label = QLabel(label_text)
            label.setStyleSheet("color: #bdb8e3; font-size: 11px;")  # Küçük açıklama metni
        
            value_label.setStyleSheet(value_style or "color: #fff; font-size: 14px; font-weight: bold;")  # Değer net ve belirgin
        
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(value_label)
            system_info_layout.addLayout(row)
        
        # Dinamik değer alanları
        self.working_time_label = QLabel()
        self.total_treatment_label = QLabel("0 seans")
        
        # Bilgi satırları
        add_info_row("🔢 Yazılım Sürümü:", QLabel("v2.1.4"), "color: #ffffff; font-size: 10px; font-weight: bold;")
        add_info_row("💻 Donanım Sürümü:", QLabel("HW-2023.1"), "color: #ffffff; font-size: 10px; font-weight: bold;")
        add_info_row("📅 Son Güncelleme:", QLabel("15 Mayıs 2025"), "color: #ffffff; font-size: 10px; font-weight: bold;")
        add_info_row("🆔 Cihaz ID:", QLabel("PEMF-001-2025"), "color: #ffffff; font-size: 10px; font-weight: bold;")
        add_info_row("⏳ Çalışma Süresi:", self.working_time_label, "color: #ffd86b; font-size: 10px; font-weight: bold;")
        add_info_row("💉 Toplam Tedavi:", self.total_treatment_label, "color: #ffd86b; font-size: 10px; font-weight: bold;")
        
        info_layout.addWidget(system_info_card)
        
        # Çalışma süresi güncelle
        self.update_working_time_label()
        
        info_layout.addStretch(1)
        content_layout.addWidget(info_panel, stretch=1)

        # --- Working Time Persistence ---
        # Timer to increment working time
        self.working_timer = QTimer(self)
        self.working_timer.timeout.connect(self.increment_working_time)
        self.working_timer.start(1000)
        
        
        

    def update_clock(self):
        from datetime import datetime
        try:
            now = datetime.now()
            self.clock.setText(f"\u23F0 {now.strftime('%d/%m/%Y %H:%M:%S')}")
        except Exception as e:
            pass

    def input_button_clicked(self):
        sender = self.sender()
        button_text = sender.text()
        print(f"Button clicked: {button_text}")
        
        # Bobin Kapat butonuna basıldığında tüm PWM'leri durdur
        if button_text == "Bobin Kapat!" and self.system_running:
            self.stop_all_pwm()
    
    def stop_all_pwm(self):
        """Stop PWM generation for all channels"""
        # Signal Generator penceresi açıksa
        if hasattr(self, 'signal_generator') and self.signal_generator:
            # Tüm kanallar için stop komutu gönder (1-8)
            for channel in range(1, 9):
                # Stop butonunu işaretle ve start butonunu işareti kaldır
                if hasattr(self.signal_generator, 'stop_btns') and len(self.signal_generator.stop_btns) >= channel:
                    self.signal_generator.stop_btns[channel-1].setChecked(True)
                    
                if hasattr(self.signal_generator, 'start_btns') and len(self.signal_generator.start_btns) >= channel:
                    self.signal_generator.start_btns[channel-1].setChecked(False)
                
                # PWM durdurma komutunu gönder
                self.signal_generator.publish_command(channel, "stop")
                
                # PWM durumunu güncelle
                self.signal_generator.pwm_status[channel] = False
            
            # Ayarları kaydet
            self.signal_generator.save_pwm_settings()
            self.signal_generator.update_status("All PWM channels stopped")
            
            # Kullanıcıya bilgi ver
            QMessageBox.information(self, "PWM Durduruldu", "Tüm bobinlerin PWM sinyalleri durduruldu.")
        else:
            print("Signal Generator penceresi açık değil, PWM durdurulamadı.")
            QMessageBox.warning(self, "Uyarı", "Signal Generator penceresi açık değil, PWM durdurulamadı.")

    def open_signal_generator(self):
        """Open the Signal Generator window"""
        # COM6 port is already opened by MainWindow.init_serial_port
        # Create SignalGeneratorWindow without parent to open in a separate window
        self.signal_generator = SignalGeneratorWindow()  # No parent parameter
        self.signal_generator.main_window_ref = self  # Keep reference to main window
        self.signal_generator.show()

    def open_sensor_data_window(self):
        self.sensor_data_window = SensorDataWindow()
        self.sensor_data_window.main_window_ref = self
        self.sensor_data_window.show()
        
        # Sensör veri penceresi açıldığında otomatik Excel kaydetme işlemini başlat
        # İlk açılışta bir kez kaydet
        self.sensor_data_window.save_sensor_data_to_excel()
        
    def open_AI_window(self):
        self.AI_window = AIWindow()
        self.AI_window.show()

    def get_latest_intensity(self):
        # Try to get the latest Hall Effect value from the sensor data window
        try:
            if hasattr(self, 'sensor_data_window') and self.sensor_data_window:
                # hall_data bir liste içinde deque nesneleri içeriyor, bu yüzden önce kanal seçilmeli
                # Varsayılan olarak ilk kanalı (0) kullanıyoruz
                if self.sensor_data_window.hall_data and len(self.sensor_data_window.hall_data) > 0:
                    # İlk kanalın (0) deque nesnesinin son elemanını al
                    if len(self.sensor_data_window.hall_data[0]) > 0:
                        return self.sensor_data_window.hall_data[0][-1]
        except Exception as e:
            print(f"[DEBUG] Could not get latest intensity: {e}")
        return 0.0

    def on_output1_freq_changed(self):
        # Get frequency from Output 1 freq_box
        try:
            freq_box = self.pasco_window.output1.content_area.widget().freq_box
            freq_text = freq_box.currentText().replace('Hz','').strip()
            freq = float(freq_text) if freq_text else 0.0
        except Exception as e:
            print(f"[DEBUG] Could not parse frequency: {e}")
            freq = 0.0
        intensity = self.get_latest_intensity()
        self.update_realtime_graph(freq, intensity)

    def open_digital_twin_window(self):
        exe_path = r"C:\Users\merta\PEMFV2\PEMF.exe"
        try:
            os.startfile(exe_path)
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Dosya açılamadı:\n{e}")
        

    def open_kpi_dashboard(self):
        self.kpi_dashboard_window = KPIDashboardWindow(main_window=self)
        self.kpi_dashboard_window.show()

    def open_autonomous_mode(self):
        self.autonomous_mode_window = AutonomousModeWindow()
        self.autonomous_mode_window.show()

    def toggle_coil(self, channel):
        """Toggle active state for a coil and update its visibility.
        
        Args:
            channel (int): Coil number (1-8)
        """
        if channel in self.active_coils:
            self.active_coils.remove(channel)
            # Hide the curves when coil is deactivated
            self.freq_curves[channel].hide()
            self.intensity_curves[channel].hide()
        else:
            self.active_coils.add(channel)
            # Show the curves when coil is activated
            self.freq_curves[channel].show()
            self.intensity_curves[channel].show()
        
        # Update the legend to reflect the current state
        self.realtime_graph.legend.items = []  # Clear existing legend items
        self.realtime_graph.legend = None  # Remove the old legend
        
        # Create a new legend with only active coils
        legend = self.realtime_graph.addLegend(offset=(10, 10))
        for coil in sorted(self.active_coils):
            legend.addItem(self.freq_curves[coil], f'Bobin {coil} Frekans')
            legend.addItem(self.intensity_curves[coil], f'Bobin {coil} Yoğunluk')

    def update_realtime_graph(self, freq, intensity, channel=1):
        """Update the real-time graph with new frequency and intensity data for a specific channel.
        
        Args:
            freq (float): Frequency value in Hz
            intensity (float): Magnetic field intensity in mT
            channel (int): Coil number (1-8)
        """
        import time
        now = time.time()
        
        # Initialize graph start time if this is the first update
        if self.graph_start_time is None:
            self.graph_start_time = now
        
        # Calculate time since start
        t = now - self.graph_start_time
        
        # Add current time to time data if it's a new timestamp
        if not self.graph_time_data or t > self.graph_time_data[-1]:
            self.graph_time_data.append(t)
        
        # Ensure the channel exists in our data structures
        if channel not in self.graph_freq_data:
            self.graph_freq_data[channel] = []
        if channel not in self.graph_intensity_data:
            self.graph_intensity_data[channel] = []
        
        # Add new data points
        self.graph_freq_data[channel].append(freq)
        self.graph_intensity_data[channel].append(intensity)
        
        # Keep only the last 300 points for performance
        max_points = 300
        if len(self.graph_time_data) > max_points:
            self.graph_time_data = self.graph_time_data[-max_points:]
        
        # Trim data for the current channel
        if len(self.graph_freq_data[channel]) > max_points:
            self.graph_freq_data[channel] = self.graph_freq_data[channel][-max_points:]
            self.graph_intensity_data[channel] = self.graph_intensity_data[channel][-max_points:]
        
        # Only update curves for active coils
        if channel in self.active_coils:
            # Get the time data that matches the length of our channel data
            time_data = self.graph_time_data[-len(self.graph_freq_data[channel]):]
            
            # Update the curves with the new data
            self.freq_curves[channel].setData(time_data, self.graph_freq_data[channel])
            self.intensity_curves[channel].setData(time_data, self.graph_intensity_data[channel])
            
            # Otomatik ölçeklendirme özelliği kaldırıldı

    def increment_working_time(self):
        self.working_seconds += 1
        try:
            with open(self.working_time_file, 'w') as f:
                f.write(str(self.working_seconds))
        except Exception:
            pass
        self.update_working_time_label()

    def update_working_time_label(self):
        s = self.working_seconds
        hours = s // 3600
        minutes = (s % 3600) // 60
        seconds = s % 60
        self.working_time_label.setText(f"{hours} saat {minutes} dakika {seconds} saniye")

    def update_kpi_effectiveness(self, value):
        self.kpi1_value.setText(f"<span style='color:#22c55e; font-size: 16px; font-weight: bold;'>{value:.1f}%</span>")
    def update_kpi_energy(self, value):
        self.kpi2_value.setText(f"<span style='color:#eab308; font-size: 16px; font-weight: bold;'>{value:.2f} kWh/tedavi</span>")
    def update_kpi_operation_rate(self, value):
        self.kpi3_value.setText(f"<span style='color:#2563eb; font-size: 16px; font-weight: bold;'>{value:.1f}%</span>")



class KPIDashboardWindow(QMainWindow):
    def __init__(self, main_window=None):
        super().__init__()
        self.setWindowTitle("KPI Dashboard")
        self.setMinimumSize(800, 600)  # Set minimum size instead of fixed geometry
        self.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        self.main_window = main_window
        
        # Create main widget and layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)
        
        # Title
        title = QLabel("<b style='font-size:28px;color:#fff;'>KPI Dashboard</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # KPI data
        kpis = [
            ("Treatment Effectiveness Rate (%)", "Measurement of patient recovery rate after treatment!", "Evaluate the biological effectiveness of the treatment and increase the success rate."),
            ("Average Treatment Duration (min)", "Average treatment duration applied in a session.", "Increase efficiency by optimizing treatment duration."),
            ("Energy Consumption Efficiency (kWh/treatment)", "Average amount of energy consumed per treatment!", "Monitor energy efficiency and optimize costs."),
            ("Device Operation Rate (%)", "Active rate in the total operation time of the device!", "Measure the usability and reliability of the device."),
            ("Patient Satisfaction Score (1-10)", "Satisfaction score given by patients after treatment!", "Increase user satisfaction and collect feedback."),
            ("Maintenance and Repair Time (min/day)", "Total duration of planned stops!", "To ensure continuity by analyzing failures and reasons for stops.")
        ]
        
        # Create scroll area for KPI cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        # Container for the grid
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(16)
        
        # Input widgets list
        self.input_widgets = []
        
        # Create KPI cards
        for i, (name, definition, purpose) in enumerate(kpis):
            # Create card
            card = QWidget()
            card.setStyleSheet("""
                QWidget {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #a98fdc, stop:1 #e0d6f7);
                    border-radius: 12px;
                    border: 1px solid #bba0e3;
                }
            """)
            
            # Card layout
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 16, 16, 16)
            card_layout.setSpacing(12)
            
            # Top row with KPI name and input
            top_row = QHBoxLayout()
            top_row.setSpacing(12)
            
            # KPI name with word wrap
            kpi_name = QLabel(f"<b style='font-size:18px;color:#3b82f6;'>{name}</b>")
            kpi_name.setWordWrap(True)
            kpi_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            top_row.addWidget(kpi_name, 3)  # 3/4 of the width
            
            # Input field
            input_container = QWidget()
            input_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            input_layout = QHBoxLayout(input_container)
            input_layout.setContentsMargins(0, 0, 0, 0)
            input_layout.setSpacing(0)
            
            if i == 0:  # Treatment Effectiveness
                input_widget = QDoubleSpinBox()
                input_widget.setSuffix(" %")
                input_widget.setRange(0, 100)
                input_widget.setDecimals(1)
                input_widget.setValue(78.0)
                input_widget.setSingleStep(0.1)
                input_widget.setMinimumWidth(80)
                input_widget.setMaximumWidth(120)
                input_widget.valueChanged.connect(lambda val, mw=self.main_window: mw and mw.update_kpi_effectiveness(val))
            elif i == 1:  # Average Treatment Duration
                input_widget = QDoubleSpinBox()
                input_widget.setSuffix(" min")
                input_widget.setRange(0, 120)
                input_widget.setDecimals(1)
                input_widget.setValue(15.0)
                input_widget.setSingleStep(0.5)
                input_widget.setMinimumWidth(80)
                input_widget.setMaximumWidth(120)
            elif i == 2:  # Energy Consumption
                input_widget = QDoubleSpinBox()
                input_widget.setSuffix(" kWh")
                input_widget.setRange(0, 10)
                input_widget.setDecimals(2)
                input_widget.setValue(0.12)
                input_widget.setSingleStep(0.01)
                input_widget.setMinimumWidth(80)
                input_widget.setMaximumWidth(120)
                input_widget.valueChanged.connect(lambda val, mw=self.main_window: mw and mw.update_kpi_energy(val))
            elif i == 3:  # Device Operation Rate
                input_widget = QDoubleSpinBox()
                input_widget.setSuffix(" %")
                input_widget.setRange(0, 100)
                input_widget.setDecimals(1)
                input_widget.setValue(95.0)
                input_widget.setSingleStep(0.5)
                input_widget.setMinimumWidth(80)
                input_widget.setMaximumWidth(120)
                input_widget.valueChanged.connect(lambda val, mw=self.main_window: mw and mw.update_kpi_operation_rate(val))
            elif i == 4:  # Patient Satisfaction
                input_widget = QDoubleSpinBox()
                input_widget.setRange(1, 10)
                input_widget.setDecimals(1)
                input_widget.setValue(8.5)
                input_widget.setSingleStep(0.1)
                input_widget.setMinimumWidth(60)
                input_widget.setMaximumWidth(100)
            else:  # Maintenance Time
                input_widget = QSpinBox()
                input_widget.setSuffix(" min")
                input_widget.setRange(0, 1440)
                input_widget.setValue(30)
                input_widget.setMinimumWidth(80)
                input_widget.setMaximumWidth(120)
            
            # Style the input widget
            input_widget.setStyleSheet("""
                QSpinBox, QDoubleSpinBox, QLineEdit {
                    background: white;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 6px 8px;
                    min-height: 32px;
                    color: #1e293b;
                }
                QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
                    border-color: #93c5fd;
                }
                QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                    border-color: #3b82f6;
                    border-width: 2px;
                }
            """)
            
            input_layout.addWidget(input_widget)
            top_row.addWidget(input_container, 1)  # 1/4 of the width
            card_layout.addLayout(top_row)
            
            # KPI definition
            kpi_def = QLabel(f"<span style='font-size:14px;color:#4b5563;'>{definition}</span>")
            kpi_def.setWordWrap(True)
            card_layout.addWidget(kpi_def)
            
            # KPI purpose
            kpi_purpose = QLabel(f"<span style='font-size:14px;color:#047857;'>{purpose}</span>")
            kpi_purpose.setWordWrap(True)
            card_layout.addWidget(kpi_purpose)
            
            # Store the input widget
            self.input_widgets.append(input_widget)
            
            # Add card to grid (2 columns, 3 rows)
            row = i // 2
            col = i % 2
            grid.addWidget(card, row, col, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        
        # Set up the scroll area
        scroll.setWidget(container)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll)
        
        # Set size policies for responsive layout
        main_layout.setStretch(1, 1)  # Make scroll area expandable

class AutonomousModeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Treatment")
        self.resize(900, 700)
        self.setMinimumSize(360, 480)  # Daha küçük cihazlar için minimum makul boyut
        
        self.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d185a, stop:1 #6c2b8f);
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: white;  /* Genel beyaz yazı */
            }
            QLabel {
                color: white;  /* QLabel'lar beyaz */
            }
            QPushButton, QComboBox {
                min-height: 40px;
                padding: 6px 16px;
                border-radius: 8px;
                font-size: 15px;
                color: white;  /* Buton ve combo iç yazılar beyaz */
            }
            QComboBox {
                background: #3d206b;
                border: 1px solid #e3e8f0;
            }
        """)
        
        # Ana widget ve layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Kart container (card)
        self.card = QWidget()
        self.card.setStyleSheet("""
            background: rgba(0, 0, 0, 0.25);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 20px;
            max-width: 800px;
        """)
        
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setSpacing(14)
        
        # Başlık
        self.title_label = QLabel("Smart Treatment")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; color: white;")
        self.card_layout.addWidget(self.title_label)
        
        # Treatment Target satırı
        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        target_label = QLabel("Treatment Target:")
        target_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        target_label.setStyleSheet("font-size: 16px; color: white;")
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Pain Reduction", "Kidney Cancer Treatment"])
        self.target_combo.setStyleSheet("""
            padding: 6px 18px;
            border-radius: 8px;
            background: #3d206b;
            color: white;
            border: 1px solid #e3e8f0;
            font-size: 16px;
        """)
        self.target_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        target_row.addWidget(target_label)
        target_row.addWidget(self.target_combo)
        self.card_layout.addLayout(target_row)
        
        # Patient Profile satırı
        profile_row = QHBoxLayout()
        profile_row.setSpacing(10)
        profile_label = QLabel("Patient Profile:")
        profile_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        profile_label.setStyleSheet("font-size: 16px; color: white;")
        
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Child", "Adult", "Elderly"])
        self.profile_combo.setStyleSheet("""
            padding: 6px 18px;
            border-radius: 8px;
            background: #3d206b;
            color: white;
            border: 1px solid #e3e8f0;
            font-size: 16px;
        """)
        self.profile_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self.profile_combo)
        self.card_layout.addLayout(profile_row)
        
        # Recommended Parameters başlığı
        self.param_title = QLabel("Recommended Parameters:")
        self.param_title.setStyleSheet("font-weight: bold; font-size: 17px; margin-top: 12px; color: white;")
        self.card_layout.addWidget(self.param_title)
        
        # Parametre listesi
        param_names = ["Frequency", "Duration", "Intensity", "Waveform"]
        self.param_labels = []
        param_list = QVBoxLayout()
        param_list.setSpacing(6)
        for name in param_names:
            label = QLabel(f"- {name}: ")
            label.setStyleSheet("font-size: 15px; color: white;")
            param_list.addWidget(label)
            self.param_labels.append(label)
        self.card_layout.addLayout(param_list)
        
        # Butonlar satırı
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        self.confirm_btn = QPushButton("Confirm and Start")
        self.confirm_btn.setStyleSheet("""
            background: #22c55e;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 24px;
        """)
        self.confirm_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.manual_btn = QPushButton("Make Manual Adjustments")
        self.manual_btn.setStyleSheet("""
            background: white;
            color: #3b82f6;
            border: 2px solid #3b82f6;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 24px;
        """)
        self.manual_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.back_btn = QPushButton("Back")
        self.back_btn.setStyleSheet("""
            background: #475569;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 24px;
        """)
        self.back_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        btn_row.addWidget(self.confirm_btn)
        btn_row.addWidget(self.manual_btn)
        btn_row.addWidget(self.back_btn)
        self.card_layout.addLayout(btn_row)
        
        # Card container ortalama (yatayda ortala)
        card_container = QHBoxLayout()
        card_container.addStretch(1)
        card_container.addWidget(self.card)
        card_container.addStretch(1)
        
        main_layout.addLayout(card_container)
        main_layout.addStretch(1)
        
        # Fontları pencere boyutuna göre ayarla
        self.update_fonts()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_fonts()
    
    def update_fonts(self):
        w = self.width()
        h = self.height()
        
        title_size = max(16, min(w // 30, h // 20))
        label_size = max(12, min(w // 45, h // 30))
        param_size = max(11, min(w // 50, h // 35))
        btn_size = max(12, min(w // 40, h // 25))
        
        title_font = QFont("Segoe UI", title_size, QFont.Weight.Bold)
        label_font = QFont("Segoe UI", label_size)
        param_font = QFont("Segoe UI", param_size)
        btn_font = QFont("Segoe UI", btn_size, QFont.Weight.Bold)
        
        self.title_label.setFont(title_font)
        self.param_title.setFont(QFont("Segoe UI", label_size, QFont.Weight.Bold))
        
        for widget in [self.target_combo, self.profile_combo]:
            widget.setFont(label_font)
        
        for lbl in self.param_labels:
            lbl.setFont(param_font)
        
        for btn in [self.confirm_btn, self.manual_btn, self.back_btn]:
            btn.setFont(btn_font)


def main():
    app = QApplication(sys.argv)
    
    # Set application icon (using the new heart with EMF icon)
    app_icon = QIcon('pemf_heart_emf_icon.ico')
    app.setWindowIcon(app_icon)
    
    window = MainWindow()
    window.setWindowIcon(app_icon)  # Also set icon for the main window
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
