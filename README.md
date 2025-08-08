# Image Redaction Tool

A powerful web-based image redaction tool that automatically detects and redacts sensitive information from images using advanced AI and machine learning techniques.

## 🚀 Features

### Core Capabilities
- **🔍 Facial Recognition**: Advanced AI-powered face detection and automatic blurring
- **🚗 License Plate Detection**: Intelligent detection and redaction using YOLO and OCR
- **📝 Text Recognition**: OCR-based text detection and redaction
- **📦 Batch Processing**: Process multiple images simultaneously
- **⚡ Real-time Processing**: Fast, efficient processing with progress tracking

### Technical Features
- **Web-based Interface**: Modern, responsive design
- **RESTful API**: Clean API endpoints for integration
- **Multi-threaded Processing**: Optimized for performance
- **Docker Support**: Easy deployment and scaling
- **Cloud Ready**: Optimized for Render.com deployment

## 🛠️ Technology Stack

### AI/ML Libraries
- **MediaPipe**: Face detection and landmarks
- **YOLO v8**: Object detection for license plates
- **EasyOCR**: Optical character recognition
- **OpenCV**: Computer vision and image processing

### Web Framework
- **Flask**: Python web framework
- **Bootstrap 5**: Responsive UI components
- **JavaScript**: Interactive frontend functionality

### Infrastructure
- **Docker**: Containerization
- **Gunicorn**: WSGI HTTP Server
- **Render.com**: Cloud deployment platform

## 📋 Prerequisites

- Python 3.10 or higher
- Docker (for containerized deployment)
- Git

## 🔧 Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/image-redaction-tool.git
cd image-redaction-tool
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create necessary directories**
```bash
mkdir uploads processed
```

5. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t image-redaction-tool .
```

2. **Run the container**
```bash
docker run -p 5000:5000 image-redaction-tool
```

## 🌐 Deployment on Render.com

### Automatic Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Render.com**
   - Create a Render.com account
   - Connect your GitHub repository
   - Choose "Web Service" deployment

3. **Configure deployment**
   - Render will automatically detect the `render.yaml` configuration
   - The app will deploy using Docker
   - Choose your preferred region and plan

4. **Environment Variables**
   - `SECRET_KEY`: Automatically generated
   - `FLASK_ENV`: Set to `production`
   - `PORT`: Set to `5000`

### Manual Deployment

1. **Create a new Web Service** on Render.com

2. **Configure the service**
   - **Environment**: Docker
   - **Build Command**: `docker build -t app .`
   - **Start Command**: `docker run -p $PORT:5000 app`

3. **Set environment variables**
   - Add required environment variables in Render dashboard

## 📖 Usage

### Single Image Processing

1. **Upload Image**
   - Drag and drop an image or click to browse
   - Supported formats: JPG, PNG, BMP, TIFF, GIF
   - Maximum file size: 16MB

2. **Configure Options**
   - ✅ Blur Faces: Enable facial detection and blurring
   - ✅ Redact License Plates: Enable license plate detection
   - ✅ Redact Text: Enable text detection and redaction
   - 🎛️ Blur Intensity: Adjust blur strength (5-50)

3. **Process & Download**
   - Click "Process Image"
   - Monitor real-time progress
   - Download the redacted image

### Batch Processing

1. **Upload Multiple Images**
   - Select multiple image files
   - Review the file list
   - Remove unwanted files if needed

2. **Configure Batch Options**
   - Same options as single image processing
   - Applied to all selected images

3. **Process & Download**
   - Start batch processing
   - Monitor progress with statistics
   - Download all results as ZIP file

## 🔌 API Endpoints

### Upload Single Image
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Image file
```

### Process Single Image
```http
POST /process
Content-Type: application/json

Body:
{
  "filename": "uploaded_filename",
  "options": {
    "blur_faces": true,
    "redact_plates": true,
    "redact_text": true,
    "blur_intensity": 15
  }
}
```

### Check Processing Status
```http
GET /status/{process_id}
```

### Download Result
```http
GET /download/{filename}
```

### Batch Upload
```http
POST /batch_upload
Content-Type: multipart/form-data

Parameters:
- files: Multiple image files
```

### Batch Process
```http
POST /batch_process
Content-Type: application/json

Body:
{
  "filenames": ["file1.jpg", "file2.png"],
  "options": {
    "blur_faces": true,
    "redact_plates": true,
    "redact_text": true,
    "blur_intensity": 15
  }
}
```

## ⚙️ Configuration

### Model Settings

The tool uses pre-trained models that are automatically downloaded on first use:

- **MediaPipe Face Detection**: Built-in model
- **YOLO v8**: Downloads `yolov8n.pt` (~6MB)
- **EasyOCR**: Downloads language models as needed

### Performance Tuning

#### For Local Development
```python
# Adjust worker count based on CPU cores
workers = multiprocessing.cpu_count()

# Enable GPU acceleration (if available)
ocr_reader = easyocr.Reader(['en'], gpu=True)
```

#### For Production
```python
# Configure gunicorn workers in Dockerfile
CMD gunicorn --workers 4 --timeout 300 app:app
```

## 🔒 Security Considerations

### Data Privacy
- Images are temporarily stored during processing
- Files are automatically cleaned up after processing
- No data is permanently stored on the server

### Input Validation
- File type validation (images only)
- File size limits (16MB max)
- Secure filename handling

### Production Security
- Use environment variables for sensitive configuration
- Enable HTTPS in production
- Implement rate limiting for API endpoints

## 🐛 Troubleshooting

### Common Issues

1. **Models not downloading**
   - Check internet connection
   - Ensure sufficient disk space
   - Verify Python package installations

2. **Memory issues with large images**
   - Reduce image size before processing
   - Increase system memory allocation
   - Use smaller batch sizes

3. **Slow processing**
   - Enable GPU acceleration if available
   - Reduce blur intensity for faster processing
   - Use smaller image sizes

### Debug Mode

Enable debug mode for development:

```bash
export FLASK_ENV=development
python app.py
```

## 📊 Performance Benchmarks

### Processing Times (Approximate)
- **Single Image (1920x1080)**: 3-8 seconds
- **Batch (10 images)**: 30-60 seconds
- **Face Detection**: ~1-2 seconds per image
- **License Plate Detection**: ~2-3 seconds per image
- **Text Detection**: ~2-4 seconds per image

### Resource Usage
- **Memory**: 1-2GB for typical workloads
- **CPU**: Multi-core utilization for batch processing
- **Storage**: Temporary files, auto-cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the API documentation

## 🚀 Deployment Status

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 📈 Roadmap

### Upcoming Features
- [ ] Additional blur/redaction methods
- [ ] Custom object detection models
- [ ] API authentication
- [ ] Webhook notifications
- [ ] Advanced batch processing options
- [ ] Export to different formats
- [ ] Integration with cloud storage

### Version History
- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added batch processing
- **v1.2.0**: Docker support and Render.com deployment
- **v1.3.0**: Enhanced UI and performance improvements

---

**Made with ❤️ for privacy and security**