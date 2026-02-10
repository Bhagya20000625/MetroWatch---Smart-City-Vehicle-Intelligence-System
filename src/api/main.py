"""
FastAPI Main Application
Entry point for the MetroWatch Vehicle Analytics API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

# Create FastAPI app
app = FastAPI(
    title="MetroWatch Vehicle Analytics API",
    description="Smart City Vehicle Intelligence System - Detect vehicles, recognize license plates, and track traffic analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "MetroWatch Vehicle Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "detect": "/api/v1/detect",
            "analytics": "/api/v1/analytics/summary",
            "health": "/api/v1/health"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("=" * 80)
    print("MetroWatch Vehicle Analytics API - Starting...")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    print("=" * 80)
    print("MetroWatch Vehicle Analytics API - Shutting down...")
    print("=" * 80)
