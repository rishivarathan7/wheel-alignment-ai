from fastapi import FastAPI

app = FastAPI(
    title="AI Wheel Alignment Monitoring System",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "AI Wheel Alignment Monitoring System",
        "status": "Online",
        "platform": "Vercel"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }