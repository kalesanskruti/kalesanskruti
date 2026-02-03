from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.predictor import predictor_service
import time

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting Remaining Useful Life (RUL) of aircraft engines using NASA CMAPSS dataset.",
    version="1.0.0"
)

class EngineData(BaseModel):
    setting1: float
    setting2: float
    setting3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Predictive Maintenance API</title>
            <style>
                body {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    color: #f8fafc;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    text-align: center;
                }
                .container {
                    background: rgba(30, 41, 59, 0.7);
                    backdrop-filter: blur(10px);
                    padding: 3rem;
                    border-radius: 1.5rem;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    max-width: 600px;
                }
                h1 {
                    font-size: 2.5rem;
                    margin-bottom: 1rem;
                    background: linear-gradient(to right, #38bdf8, #818cf8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                p {
                    font-size: 1.125rem;
                    color: #94a3b8;
                    line-height: 1.6;
                }
                .status {
                    display: inline-block;
                    margin-top: 2rem;
                    padding: 0.5rem 1rem;
                    background: rgba(34, 197, 94, 0.2);
                    color: #4ade80;
                    border-radius: 9999px;
                    font-weight: 600;
                    font-size: 0.875rem;
                    border: 1px solid rgba(34, 197, 94, 0.3);
                }
                .links {
                    margin-top: 2rem;
                }
                a {
                    color: #38bdf8;
                    text-decoration: none;
                    font-weight: 500;
                    transition: color 0.2s;
                }
                a:hover {
                    color: #7dd3fc;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Predictive Maintenance</h1>
                <p>Advanced failure detection and RUL prediction system powered by NASA's CMAPSS dataset and Random Forest Regressor.</p>
                <div class="status">System Online</div>
                <div class="links">
                    <a href="/docs">View API Documentation</a>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict")
async def predict(data: List[EngineData]):
    # Convert Pydantic models to dicts
    input_data = [item.dict() for item in data]
    
    result = predictor_service.predict(input_data)
    
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {"predictions": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
