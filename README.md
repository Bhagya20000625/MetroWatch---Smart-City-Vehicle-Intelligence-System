# 🚗 MetroWatch - Smart City Vehicle Intelligence System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent vehicle monitoring and analytics platform designed for smart city infrastructure. MetroWatch combines advanced computer vision, license plate recognition, real-time tracking, and comprehensive analytics to provide actionable insights for traffic management.

## 🌟 Features

### Core Capabilities
- **🚙 Multi-Vehicle Detection**: Real-time detection of cars, motorcycles, buses, and trucks using YOLOv8
- **🔍 License Plate Recognition**: Custom-trained OCR system for Sri Lankan vehicle plates
- **🗺️ Province Detection**: Automatic identification of vehicle origin by province codes
- **📹 Real-Time Tracking**: SORT algorithm with Kalman filtering for persistent vehicle tracking
- **📊 Entry/Exit Analytics**: Zone-based counting for traffic flow analysis
- **💾 Data Persistence**: PostgreSQL database for historical analytics and reporting
- **🌐 REST API**: FastAPI-powered backend with comprehensive endpoints
- **📈 Interactive Dashboard**: Real-time visualization of traffic patterns and statistics

### Technical Highlights
- **GPU Acceleration**: CUDA support for high-performance inference
- **Custom Models**: Fine-tuned license plate detector trained on 7,057+ images
- **Scalable Architecture**: Microservices-ready design with Docker support
- **Database Migrations**: Alembic-managed schema versioning
- **Production Ready**: Environment-based configuration and error handling

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MetroWatch System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   Frontend   │ ───> │  FastAPI     │ ───> │ PostgreSQL   │ │
│  │   Dashboard  │ <─── │   Backend    │ <─── │   Database   │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│                               │                                  │
│                               ↓                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │              Vehicle Analytics Service                      ││
│  │  ┌───────────┐  ┌──────────┐  ┌────────┐  ┌─────────────┐││
│  │  │  YOLOv8   │→│ Plate    │→│Province│→│   SORT      │││
│  │  │ Detection │ │   OCR    │ │Detector│ │  Tracking   │││
│  │  └───────────┘  └──────────┘  └────────┘  └─────────────┘││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Machine Learning & Computer Vision
- **YOLOv8** (Ultralytics): Vehicle detection and custom plate detection
- **EasyOCR**: License plate text recognition
- **OpenCV**: Image processing and video handling
- **NumPy**: Numerical computations

### Tracking & Analytics
- **SORT Algorithm**: Multi-object tracking
- **Kalman Filter** (FilterPy): Position prediction and smoothing
- **Hungarian Algorithm** (LAP): Object association
- **SciPy**: Scientific computing utilities

### Backend & API
- **FastAPI**: High-performance async API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and serialization
- **Python-Multipart**: File upload handling

### Database & ORM
- **PostgreSQL 16**: Relational database
- **SQLAlchemy 2.0**: ORM and query builder
- **Alembic**: Database migration management
- **Psycopg2**: PostgreSQL adapter

### Development & Deployment
- **Python 3.12**: Core language
- **Python-dotenv**: Environment management
- **Git**: Version control
- **CUDA/cuDNN**: GPU acceleration (optional)

---

## 📋 Prerequisites

### Required
- **Python**: 3.12 or higher
- **PostgreSQL**: 16.x or higher
- **Git**: For version control
- **pip**: Python package manager

### Recommended
- **NVIDIA GPU**: RTX 3050 or better (6GB+ VRAM)
- **CUDA Toolkit**: 11.8+ (for GPU acceleration)
- **16GB RAM**: For processing high-resolution videos
- **SSD Storage**: For faster model loading

### Operating System
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System.git
cd MetroWatch---Smart-City-Vehicle-Intelligence-System
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (Windows with CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Set Up PostgreSQL Database
```bash
# Install PostgreSQL 16 from https://www.postgresql.org/download/

# Create database
psql -U postgres
CREATE DATABASE metrowatch;
\q
```

### 5. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and update with your settings:
# - DATABASE_URL: PostgreSQL connection string
# - API_HOST: API server host (default: 0.0.0.0)
# - API_PORT: API server port (default: 8000)
```

**Example `.env` configuration:**
```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/metrowatch
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

**Note**: If your PostgreSQL password contains special characters like `@`, URL-encode them:
- `@` → `%40`
- `#` → `%23`
- `$` → `%24`

### 6. Initialize Database
```bash
# Create database tables
python init_db.py
```

### 7. Download Pre-trained Models
Models are tracked with Git LFS or can be downloaded separately:

- **YOLOv8 Vehicle Detection**: `yolov8n.pt`, `yolov8x.pt` (auto-downloaded on first run)
- **Custom Plate Detector**: `models/license_plate_detector.pt` (train or use provided)

---

## ⚙️ Configuration

### Database Configuration
Edit `DATABASE_URL` in `.env`:
```env
DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[database]
```

### Model Configuration
Models are located in:
- `yolov8*.pt`: Root directory (vehicle detection)
- `models/license_plate_detector.pt`: Custom plate detector

### API Configuration
Modify `src/api/main.py` for CORS, middleware, or routes customization.

---

## 💻 Usage

### Starting the API Server
```bash
# Development mode (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation
Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Available Endpoints

#### 1. Health Check
```bash
GET /api/v1/health
```
Returns API status and version information.

#### 2. Vehicle Detection
```bash
POST /api/v1/detect
Content-Type: multipart/form-data

Parameters:
  - file: Image file (JPEG, PNG)
```
Detects vehicles, license plates, and provinces in uploaded image.

**Response:**
```json
{
  "detections": [
    {
      "id": 1,
      "type": "car",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 150, "x2": 400, "y2": 350},
      "plate_text": "WP ABC-1234",
      "province": "Western Province",
      "timestamp": "2026-02-10T10:30:00"
    }
  ],
  "count": 1
}
```

#### 3. Analytics Summary
```bash
GET /api/v1/analytics/summary
```
Returns comprehensive traffic analytics.

**Response:**
```json
{
  "total_vehicles": 1523,
  "total_entries": 856,
  "total_exits": 667,
  "current_count": 189,
  "by_type": {
    "car": 1120,
    "motorcycle": 245,
    "bus": 98,
    "truck": 60
  },
  "by_province": {
    "Western Province": 845,
    "Central Province": 312,
    "Southern Province": 198
  }
}
```

#### 4. Recent Vehicle History
```bash
GET /api/v1/analytics/vehicles?limit=10
```
Returns most recent vehicle detections.

#### 5. Reset Analytics
```bash
POST /api/v1/analytics/reset
```
Clears all vehicle records from database.

### Testing the API
```bash
# Run integration tests
python test_db_integration.py

# Test with sample image
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Video Processing
Use the video tracking script for batch processing:
```bash
python tests/test_video_tracking.py
```

### Database Migrations
```bash
# Generate new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

---

## 🗄️ Database Schema

### `vehicles` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Unique vehicle detection ID |
| `track_id` | Integer | SORT tracking ID for correlation |
| `vehicle_type` | String(50) | Vehicle category (car, motorcycle, etc.) |
| `confidence` | Float | Detection confidence score |
| `bbox` | JSON | Bounding box coordinates |
| `plate_text` | String(50) | Recognized license plate text |
| `province` | String(100) | Detected province name |
| `timestamp` | DateTime | Detection timestamp (UTC) |

### `vehicle_logs` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Unique log entry ID |
| `vehicle_id` | Integer (FK) | Reference to vehicles table |
| `track_id` | Integer | SORT tracking ID |
| `position_x` | Float | Vehicle center X coordinate |
| `position_y` | Float | Vehicle center Y coordinate |
| `frame_number` | Integer | Video frame number |
| `event_type` | String(20) | Event type (entry, exit, tracked) |
| `timestamp` | DateTime | Event timestamp (UTC) |

### Relationships
- `vehicles.logs`: One-to-Many relationship with `vehicle_logs`
- `vehicle_logs.vehicle`: Many-to-One relationship with `vehicles`

---

## 📁 Project Structure

```
Metro_Watch/
├── 📂 alembic/                    # Database migrations
│   ├── versions/                  # Migration scripts
│   ├── env.py                     # Alembic environment
│   └── script.py.mako            # Migration template
├── 📂 data/                       # Data storage
│   ├── images/                    # Test images
│   └── videos/                    # Test videos
├── 📂 models/                     # Trained models
│   └── license_plate_detector.pt  # Custom plate detector
├── 📂 src/                        # Source code
│   ├── 📂 api/                    # FastAPI backend
│   │   ├── main.py               # API entry point
│   │   ├── routes.py             # Endpoint definitions
│   │   ├── services.py           # Business logic
│   │   └── models.py             # Pydantic schemas
│   ├── 📂 database/               # Database layer
│   │   ├── database.py           # Connection management
│   │   └── models.py             # SQLAlchemy models
│   ├── 📂 detection/              # Vehicle detection
│   │   └── vehicle_detector.py   # YOLOv8 wrapper
│   ├── 📂 ocr/                    # License plate OCR
│   │   └── plate_recognizer.py   # EasyOCR + custom detector
│   ├── 📂 tracking/               # Vehicle tracking
│   │   └── vehicle_tracker.py    # SORT implementation
│   └── 📂 utils/                  # Utilities
│       └── province_detector.py   # Province mapping
├── 📂 tests/                      # Test scripts
│   ├── test_video_tracking.py    # Video processing demo
│   └── test_full_pipeline.py     # End-to-end testing
├── 📄 .env.example                # Environment template
├── 📄 .gitignore                 # Git ignore rules
├── 📄 .gitattributes             # Git line ending rules
├── 📄 alembic.ini                # Alembic configuration
├── 📄 init_db.py                 # Database initialization
├── 📄 requirements.txt           # Python dependencies
├── 📄 test_db_integration.py     # API integration tests
├── 📄 train_plate_detector.py    # Model training script
└── 📄 README.md                  # This file
```

---

## 🎯 Performance Considerations

### Model Performance
- **YOLOv8n**: ~120 FPS (GPU), ~8 FPS (CPU) - Lightweight, fast
- **YOLOv8x**: ~30 FPS (GPU), ~2 FPS (CPU) - High accuracy
- **Custom Plate Detector**: ~150 FPS on RTX 3050
- **OCR Processing**: ~50ms per plate on CPU

### Optimization Tips
1. **Use GPU**: Install CUDA-enabled PyTorch for 10-50x speedup
2. **Batch Processing**: Process multiple frames together
3. **Model Selection**: Use YOLOv8n for real-time, YOLOv8x for accuracy
4. **Resolution**: Lower resolution (640px) for faster processing
5. **Database Indexing**: Ensure indexes on timestamp, plate_text, vehicle_type

### Scalability
- **Horizontal Scaling**: Deploy multiple API instances behind load balancer
- **Database Pooling**: Configure connection pooling in `database.py`
- **Caching**: Implement Redis for frequently accessed analytics
- **Async Processing**: Use Celery for background video processing

---

## ⚠️ Known Limitations

### OCR Accuracy
- **Current Accuracy**: ~60-70% for Sri Lankan plates
- **Common Issues**:
  - Character confusion (W↔M, O↔0, 9↔2)
  - Missing characters in low-light conditions
  - Partial plate detections
- **Improvements**: Consider commercial OCR APIs (Google Vision, Plate Recognizer)

### Model Constraints
- **Training Data**: 7,057 images (more data needed for edge cases)
- **Weather Conditions**: Performance degrades in rain, fog, night
- **Plate Variations**: Best with standard Sri Lankan format (AB XYZ-1234)

### System Requirements
- **Memory**: High-resolution video processing requires 8-16GB RAM
- **Storage**: Video files and model weights can consume significant space
- **GPU**: CPU-only processing is significantly slower

---

## 🚀 Future Enhancements

### Short Term
- [ ] Implement Redis caching for analytics
- [ ] Add background job processing with Celery
- [ ] Create Docker containerization
- [ ] Implement unit tests with pytest
- [ ] Add logging with ELK stack

### Long Term
- [ ] Multi-camera support with camera management
- [ ] Real-time video stream processing (RTSP)
- [ ] Advanced analytics (speed detection, traffic violations)
- [ ] Mobile application (React Native)
- [ ] AI-powered incident detection
- [ ] Export reports (PDF, Excel)
- [ ] Integration with smart city platforms
- [ ] Multi-language support

### Research Directions
- [ ] Transformer-based OCR (CLIP, TrOCR)
- [ ] DeepSORT tracking for improved accuracy
- [ ] Vehicle re-identification across cameras
- [ ] Anomaly detection for suspicious behavior
- [ ] Edge deployment (NVIDIA Jetson, Raspberry Pi)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open Pull Request**

### Coding Standards
- Follow PEP 8 style guide
- Add docstrings to functions and classes
- Write unit tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies
- **Ultralytics YOLOv8**: State-of-the-art object detection
- **EasyOCR**: Multi-language OCR framework
- **FastAPI**: Modern Python web framework
- **PostgreSQL**: Powerful open-source database

### Datasets
- **Roboflow**: License plate detection dataset (7,057 images)
- Community contributions for Sri Lankan plate samples

### Inspiration
- Smart city initiatives worldwide
- Traffic management research
- Open-source computer vision community

---

## 👤 Author

**Bhagya Jayawardhana**
- GitHub: [@Bhagya20000625](https://github.com/Bhagya20000625)
- Email: [your.email@example.com]
- LinkedIn: [Your LinkedIn Profile]

---

## 📞 Support

For issues, questions, or suggestions:
- **GitHub Issues**: [Create an issue](https://github.com/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System/issues)
- **Email**: [your.email@example.com]
- **Documentation**: [Wiki](https://github.com/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System/wiki)

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Bhagya20000625/MetroWatch---Smart-City-Vehicle-Intelligence-System?style=social)

---

<div align="center">
  <strong>Built with ❤️ for Smart Cities</strong>
  <br>
  <sub>Making urban transportation safer and more efficient</sub>
</div>