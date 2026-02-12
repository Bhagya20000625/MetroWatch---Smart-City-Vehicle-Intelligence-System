"""
API Routes - FastAPI Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import DetectionResponse, AnalyticsSummary, VehicleDetection, HealthResponse
from .services import VehicleAnalyticsService
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["Vehicle Analytics"])

# Initialize service (singleton)
print("Starting Vehicle Analytics Service...")
analytics_service = VehicleAnalyticsService()


@router.post("/detect", response_model=DetectionResponse)
async def detect_vehicles(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        
        # Process with analytics service
        vehicles = analytics_service.process_frame(contents)
        
        return {
            'success': True,
            'vehicle_count': len(vehicles),
            'vehicles': vehicles,
            'timestamp': datetime.now().isoformat()
        }
    except ValueError as e:
        print(f"ValueError in detect_vehicles: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"ERROR in detect_vehicles: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@router.post("/detect/video")
async def detect_vehicles_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video (MP4, AVI, MOV)")
        
        # Read video
        contents = await file.read()
        
        print(f"Processing video: {file.filename} ({len(contents)} bytes)")
        
        # Process with analytics service
        result = analytics_service.process_video(contents)
        
        return {
            'success': True,
            'message': 'Video processed successfully',
            **result,
            'timestamp': datetime.now().isoformat()
        }
    except ValueError as e:
        print(f"ValueError in detect_vehicles_video: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"ERROR in detect_vehicles_video: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary():
    try:
        return analytics_service.get_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/vehicles")
async def get_recent_vehicles(limit: int = 10):
    try:
        return {
            'success': True,
            'vehicles': analytics_service.get_recent_vehicles(limit)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/reset")
async def reset_analytics():
    try:
        return analytics_service.reset_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
