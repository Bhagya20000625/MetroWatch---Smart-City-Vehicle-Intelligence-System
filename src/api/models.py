"""
Pydantic Models for API Request/Response Schemas
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class VehicleDetection(BaseModel):
    """Single vehicle detection result"""
    type: str
    confidence: float
    bbox: List[int]
    plate_text: Optional[str] = None
    province: Optional[str] = None


class DetectionResponse(BaseModel):
    """Response for vehicle detection endpoint"""
    success: bool
    vehicle_count: int
    vehicles: List[VehicleDetection]
    timestamp: str


class AnalyticsSummary(BaseModel):
    """Analytics summary response"""
    total_vehicles: int
    total_entries: int
    total_exits: int
    current_count: int
    by_type: dict
    by_province: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
