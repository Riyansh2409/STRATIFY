"""
Stratify FastAPI Bridge Server — Plan B (Premium Unified UI)
Bridges the React frontend ↔ Stratify pipeline & RAG components.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
from collections import namedtuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── Imports from Stratify Modules ───────────────────────────────────────────
# Add root to sys.path to allow absolute imports
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from analysis.business_charts import generate_business_charts
from analysis.pdf_exporter import generate_pdf_report
from rag.rag_pipeline import load_retriever
from rag.generator import RAGGenerator, GeneratorConfig
from rag.retriever import RAGResponse

# ── Config & Setup ──────────────────────────────────────────────────────────
UPLOAD_DIR  = BASE_DIR / "uploads"
DATA_DIR    = BASE_DIR / "processed_data"
REPORT_DIR  = BASE_DIR / "reports_output"
RAG_INDEX   = BASE_DIR / "rag_index"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stratify API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In dev, allow all for simplicity; or specific: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances for RAG
RAG_MODELS = {
    "retriever": None,
    "generator": None
}

@app.on_event("startup")
async def startup_event():
    """Load RAG models on startup for fast response."""
    try:
        if RAG_INDEX.exists() and (RAG_INDEX / "rag_meta.json").exists():
            Args = namedtuple('Args', ['index', 'backend', 'no_reranker', 'no_mmr', 'top_k'])
            args = Args(index=str(RAG_INDEX), backend="chroma", no_reranker=True, no_mmr=True, top_k=3)
            RAG_MODELS["retriever"] = load_retriever(args)
            RAG_MODELS["generator"] = RAGGenerator(GeneratorConfig(max_new_tokens=256))
            print("✓ RAG models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: RAG startup failed (usually happens before first index): {e}")

# ── Models ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    language: Optional[str] = None
    domain: Optional[str] = None

class ExportRequest(BaseModel):
    use_chi: bool = True
    use_ttest: bool = False
    use_anova: bool = False

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "mode": "Plan B (Premium)"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    # Clear previous uploads
    for existing in UPLOAD_DIR.iterdir():
        if existing.is_file(): existing.unlink()

    saved = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append({"name": file.filename, "size_kb": round(len(content)/1024, 2)})
    
    return {"status": "success", "files": saved}

@app.post("/run-pipeline")
async def run_pipeline_endpoint():
    """Runs data preprocessing AND RAG indexing."""
    if not any(UPLOAD_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        # 1. Run Preprocessing
        result = subprocess.run(
            [sys.executable, "run_pipeline.py", "--input", "uploads", "--output", "processed_data", "--report", "reports_output"],
            capture_output=True, text=True, cwd=str(BASE_DIR), timeout=300
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {result.stderr}")

        # 2. Run RAG Indexing
        data_path = DATA_DIR / "val.jsonl"
        idx_result = subprocess.run(
            [sys.executable, "-m", "rag.rag_pipeline", "index", "--data", str(data_path), "--index", str(RAG_INDEX), "--backend", "chroma", "--no-reranker"],
            capture_output=True, text=True, cwd=str(BASE_DIR), timeout=300
        )
        
        # 3. Reload models in memory
        await startup_event()

        return {"status": "success", "message": "Pipeline and indexing complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    stats_path = DATA_DIR / "pipeline_stats.json"
    if not stats_path.exists():
        return {"status": "empty"}

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    # Check for charts
    charts = {}
    figure_dir = REPORT_DIR / "figures"
    if figure_dir.exists():
        # Regenerate business charts info
        try:
            biz_charts = generate_business_charts(str(UPLOAD_DIR), str(figure_dir))
            for label, path in biz_charts.items():
                charts[label] = Path(path).name
        except: pass

    return {
        "status": "ready",
        "stats": stats,
        "charts": charts
    }

@app.get("/charts/{filename}")
def get_chart_image(filename: str):
    file_path = REPORT_DIR / "figures" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(file_path)

@app.post("/ai/recommend")
def ai_recommend():
    """Generates AI statistical test recommendations based on current stats."""
    if not RAG_MODELS["generator"]:
        raise HTTPException(status_code=503, detail="AI models not loaded. Run pipeline first.")

    stats_path = DATA_DIR / "pipeline_stats.json"
    if not stats_path.exists(): return {"recommendation": "No data available."}
    
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    prompt = f"Analyze these statistics: {json.dumps(stats.get('language_distribution', {}))}. Total chunks: {stats.get('chunks_after_filter', 0)}. Recommend Chi-Square, T-Test, or ANOVA. Explain why in 2 sentences."
    dummy_resp = RAGResponse(query=prompt, context=prompt, passages=[], sources=[])
    ans = RAG_MODELS["generator"].generate(dummy_resp)
    return {"recommendation": ans.answer}

@app.post("/ai/chat")
def ai_chat(req: ChatRequest):
    if not RAG_MODELS["retriever"] or not RAG_MODELS["generator"]:
        raise HTTPException(status_code=503, detail="RAG system not ready. Run pipeline first.")

    filters = {}
    if req.language: filters["language"] = req.language
    if req.domain: filters["domain"] = req.domain

    resp = RAG_MODELS["retriever"].retrieve(req.prompt, filters=filters if filters else None)
    ans = RAG_MODELS["generator"].generate(resp)
    
    return {
        "answer": ans.answer,
        "sources": ans.sources,
        "latency": ans.latency_sec
    }

@app.post("/export-pdf")
def export_pdf(req: ExportRequest):
    stats_path = DATA_DIR / "pipeline_stats.json"
    report_path = REPORT_DIR / "analysis_report.json"
    
    if not stats_path.exists() or not report_path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Run pipeline first.")

    with open(stats_path, "r", encoding="utf-8") as f: stats = json.load(f)
    with open(report_path, "r", encoding="utf-8") as f: report = json.load(f)

    # Simple summary call for PDF
    ai_summary = "Dataset overview based on document ingestion."
    if RAG_MODELS["generator"] and RAG_MODELS["retriever"]:
        rag_req = RAG_MODELS["retriever"].retrieve("Executive summary of document topics. 2 paragraphs.")
        ai_summary = RAG_MODELS["generator"].generate(rag_req).answer

    pdf_bytes = generate_pdf_report(stats, report, rag_summary_text=ai_summary)
    
    # Save temp and return
    temp_pdf = REPORT_DIR / "export.pdf"
    temp_pdf.write_bytes(pdf_bytes)
    return FileResponse(temp_pdf, filename="Stratify_Report.pdf", media_type="application/pdf")
