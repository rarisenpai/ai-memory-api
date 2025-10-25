from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import memory_router
from .models import HealthResponse
import os

app = FastAPI(
    title="Memory Layer API",
    description="High-level memory operations for n8n with Letta identities support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(memory_router)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {"status": "healthy", "service": "memory-layer"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "memory-layer"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)