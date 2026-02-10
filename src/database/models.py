"""
SQLAlchemy database models for vehicle tracking and analytics.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Vehicle(Base):
    """
    Vehicle detection record with license plate and province information.
    """
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    track_id = Column(Integer, nullable=True, index=True)  # SORT tracking ID
    vehicle_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)  # Store as {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    plate_text = Column(String(50), nullable=True, index=True)
    province = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship to vehicle logs
    logs = relationship("VehicleLog", back_populates="vehicle", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Vehicle(id={self.id}, type={self.vehicle_type}, plate={self.plate_text}, province={self.province})>"


class VehicleLog(Base):
    """
    Individual vehicle position log for tracking movement across frames.
    """
    __tablename__ = "vehicle_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=True, index=True)  # SORT tracking ID for correlation
    position_x = Column(Float, nullable=False)  # Center X coordinate
    position_y = Column(Float, nullable=False)  # Center Y coordinate
    frame_number = Column(Integer, nullable=True)
    event_type = Column(String(20), nullable=True, index=True)  # "entry", "exit", "tracked"
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship to vehicle
    vehicle = relationship("Vehicle", back_populates="logs")

    def __repr__(self):
        return f"<VehicleLog(id={self.id}, vehicle_id={self.vehicle_id}, event={self.event_type})>"
