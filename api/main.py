from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_system import AdvancedAnomalySystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = AdvancedAnomalySystem()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    anomalies, scores, details = system.run(df)

    return {
        "anomaly_count": int(anomalies.sum()),
        "sample_scores": scores[:20].tolist(),
        "sample_anomalies": anomalies[:20].tolist(),
        "summary": details["report"],
        "full_data": details["results"].values.tolist(),
        "full_anomalies": anomalies.tolist(),
        "full_scores": scores.tolist(),
    }