#!/usr/bin/env python3
"""
Image Redaction Web Tool
========================
A web-based image redaction tool using Flask for deployment on Render.com

Author: AI Assistant
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import easyocr
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging
import zipfile
import tempfile
from datetime import datetime
import io
import base64
from PIL import Image
import threading
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variables for models
face_detection = None
ocr_reader = None
yolo_model = None
mp_face_detection = None

# Processing status tracking
processing_status = {}

class ImageRedactor:
    def __init__(self):
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all AI models"""
        global face_detection, ocr_reader, yolo_model, mp_face_detection
        
        logger.info("Initializing models...")
        
        try:
            # MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            
            # EasyOCR for text detection
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            # YOLO for license plate detection
            try:
                yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"YOLO model failed to load: {e}")
                yolo_model = None
                
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe"""
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                faces.append((x, y, w, h))
                
        return faces
    
    def detect_license_plates(self, image):
        """Detect license plates using YOLO and OCR"""
        plates = []
        
        if yolo_model is None:
            return plates
            
        try:
            results = yolo_model(image, verbose=False)
            vehicle_classes = [2, 3, 5, 7]  # COCO dataset vehicle classes
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in vehicle_classes and conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            roi = image[y1:y2, x1:x2]
                            
                            if roi.size > 0:
                                try:
                                    ocr_results = ocr_reader.readtext(roi)
                                    for (bbox, text, confidence) in ocr_results:
                                        if confidence > 0.5 and len(text) > 3:
                                            text_bbox = bbox
                                            min_x = min([point[0] for point in text_bbox])
                                            min_y = min([point[1] for point in text_bbox])
                                            max_x = max([point[0] for point in text_bbox])
                                            max_y = max([point[1] for point in text_bbox])
                                            
                                            plates.append((
                                                x1 + int(min_x), y1 + int(min_y),
                                                int(max_x - min_x), int(max_y - min_y)
                                            ))
                                except Exception as e:
                                    logger.warning(f"OCR error: {e}")
                                    
        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
            
        return plates
    
    def detect_text_regions(self, image):
        """Detect text regions using OCR"""
        text_regions = []
        
        try:
            results = ocr_reader.readtext(image)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    min_x = min([point[0] for point in bbox])
                    min_y = min([point[1] for point in bbox])
                    max_x = max([point[0] for point in bbox])
                    max_y = max([point[1] for point in bbox])
                    
                    text_regions.append((
                        int(min_x), int(min_y),
                        int(max_x - min_x), int(max_y - min_y)
                    ))
                    
        except Exception as e:
            logger.error(f"Error in text detection: {e}")
            
        return text_regions
    
    def apply_redaction(self, image, regions, method='blur', blur_intensity=15):
        """Apply redaction to specified regions"""
        result = image.copy()
        
        for (x, y, w, h) in regions:
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if method == 'blur':
                roi = result[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)
                result[y:y+h, x:x+w] = blurred_roi
            elif method == 'black':
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 0), -1)
                
        return result
    
    def process_image(self, image_path, output_path, options):
        """Process a single image with specified options"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            result = image.copy()
            
            # Face detection and blurring
            if options.get('blur_faces', False):
                try:
                    faces = self.detect_faces_mediapipe(image)
                    result = self.apply_redaction(result, faces, 'blur', options.get('blur_intensity', 15))
                except Exception as e:
                    logger.warning(f"Face detection failed: {e}")
            
            # License plate detection and redaction
            if options.get('redact_plates', False):
                try:
                    plates = self.detect_license_plates(result)
                    result = self.apply_redaction(result, plates, 'black')
                except Exception as e:
                    logger.warning(f"License plate detection failed: {e}")
            
            # Text detection and redaction
            if options.get('redact_text', False):
                try:
                    text_regions = self.detect_text_regions(result)
                    result = self.apply_redaction(result, text_regions, 'black')
                except Exception as e:
                    logger.warning(f"Text detection failed: {e}")
            
            cv2.imwrite(output_path, result)
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False

# Initialize the redactor
redactor = ImageRedactor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 for web display"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'original_image': image_to_base64(filepath)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    filename = data.get('filename')
    options = data.get('options', {})
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Generate unique processing ID
    process_id = str(uuid.uuid4())
    
    # Start processing in background
    def background_process():
        processing_status[process_id] = {'status': 'processing', 'progress': 0}
        
        try:
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_redacted.jpg"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            
            processing_status[process_id]['progress'] = 25
            
            success = redactor.process_image(input_path, output_path, {
                'blur_faces': options.get('blur_faces', False),
                'redact_plates': options.get('redact_plates', False),
                'redact_text': options.get('redact_text', False),
                'blur_intensity': int(options.get('blur_intensity', 15))
            })
            
            processing_status[process_id]['progress'] = 75
            
            if success:
                processing_status[process_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'output_filename': output_filename,
                    'processed_image': image_to_base64(output_path)
                }
            else:
                processing_status[process_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': 'Processing failed'
                }
                
        except Exception as e:
            processing_status[process_id] = {
                'status': 'error',
                'progress': 0,
                'error': str(e)
            }
    
    thread = threading.Thread(target=background_process)
    thread.daemon = True
    thread.start()
    
    return jsonify({'process_id': process_id})

@app.route('/status/<process_id>')
def get_status(process_id):
    if process_id in processing_status:
        return jsonify(processing_status[process_id])
    else:
        return jsonify({'error': 'Process ID not found'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return "Error downloading file", 500

@app.route('/batch')
def batch_upload():
    return render_template('batch.html')

@app.route('/batch_upload', methods=['POST'])
def batch_upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filename)
    
    return jsonify({
        'success': True,
        'uploaded_files': uploaded_files,
        'count': len(uploaded_files)
    })

@app.route('/batch_process', methods=['POST'])
def batch_process():
    data = request.get_json()
    filenames = data.get('filenames', [])
    options = data.get('options', {})
    
    if not filenames:
        return jsonify({'error': 'No files provided'}), 400
    
    process_id = str(uuid.uuid4())
    
    def background_batch_process():
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'processed': 0,
            'total': len(filenames),
            'results': []
        }
        
        try:
            processed_files = []
            
            for i, filename in enumerate(filenames):
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                if os.path.exists(input_path):
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}_redacted.jpg"
                    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                    
                    success = redactor.process_image(input_path, output_path, {
                        'blur_faces': options.get('blur_faces', False),
                        'redact_plates': options.get('redact_plates', False),
                        'redact_text': options.get('redact_text', False),
                        'blur_intensity': int(options.get('blur_intensity', 15))
                    })
                    
                    if success:
                        processed_files.append(output_filename)
                
                # Update progress
                processing_status[process_id]['processed'] = i + 1
                processing_status[process_id]['progress'] = int(((i + 1) / len(filenames)) * 100)
            
            # Create zip file
            zip_filename = f"batch_redacted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(app.config['PROCESSED_FOLDER'], zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename in processed_files:
                    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
                    if os.path.exists(file_path):
                        zipf.write(file_path, filename)
            
            processing_status[process_id] = {
                'status': 'completed',
                'progress': 100,
                'processed': len(processed_files),
                'total': len(filenames),
                'zip_filename': zip_filename,
                'results': processed_files
            }
            
        except Exception as e:
            processing_status[process_id] = {
                'status': 'error',
                'progress': 0,
                'error': str(e)
            }
    
    thread = threading.Thread(target=background_batch_process)
    thread.daemon = True
    thread.start()
    
    return jsonify({'process_id': process_id})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)