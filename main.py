import torch
import gc
import os
import sys
import json
import shutil
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Generator
from pathlib import Path
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from peft import PeftModel
import faiss

from helpers import (
    get_image_embeddings,
    retrieve_top_k,
    extract_frames,
    ru_to_en,
    cluster_embeddings,
    get_centroid_images,
    build_captions_csv,
    read_captions_csv,
    request_scenarios,
    decode_response,
    prepare_pdf,
    build_hnsw_index,
    load_hnsw_index,
    run_whisper,
    load_segments,
    build_documents,
    embed_and_save_index,
    load_index,
    semantic_search,
    search_by_topic,
)

# macOS packaging support
from multiprocessing import freeze_support  # noqa

freeze_support()  # noqa


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle"""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# Logging ——————————————————————————————————————————————
log_file = resource_path("app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Check for ffmpeg ————————————————————————————————————————————————
if shutil.which("ffmpeg") is None:
    logger.critical("FFmpeg not found. Please install it and try again.")
    raise RuntimeError("FFmpeg not found. Please install it and try again.")

# SETUP ————————————————————————————————————————————————————————
# TODO - mps if on mac, code below if otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

REPORT_DIR = resource_path("./reports")
DB_DIR = resource_path("./db")

SERVER_VIDEO_DIR = "http://0.0.0.0:8000/videos"
SERVER_FRAMES_DIR = "http://0.0.0.0:8000/frames"

logger.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").to(
    device
)
clip_model = PeftModel.from_pretrained(clip_model, resource_path("models/LoRA_wedding"))
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

logger.info("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
tr_model = (
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    .to(device)
    .eval()
)


# GLOBAL STATE
image_paths, image_embeddings = [], None
whisper_store = None

# FastAPI App ———————————————————————————————————————————
app = FastAPI()


# Query classes
class FolderRequest(BaseModel):
    path: str


class AnalysisRequest(BaseModel):
    path_to_videos: str = "./videos"


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class OpenrouterRequest(BaseModel):
    openrouter_api_key: str = (
        "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"
    )
    openrouter_model: str = "qwen/qwen3-30b-a3b:free"


class WhisperRequest(BaseModel):
    model: str = "base"
    batch_size: int = 12


class WhisperSearchRequest(BaseModel):
    query: str


class WhisperTopicRequest(BaseModel):
    topic: str


def mount_folders(video_folder_path):
    global mounted

    video_folder_path = Path(video_folder_path).resolve()
    frames_path = (video_folder_path / "frames").resolve()

    resolved_video_path = None

    if not mounted["videos"]:
        resolved_video_path = video_folder_path
        app.mount(
            "/videos", StaticFiles(directory=str(resolved_video_path)), name="videos"
        )
        mounted["videos"] = True
    else:
        resolved_video_path = Path(
            app.state.VIDEO_DIR
        )  # Or reuse existing one from app.state

    if not mounted["frames"]:
        frames_path = (video_folder_path / "frames").resolve()
        if not frames_path.exists():
            frames_path.mkdir(parents=True, exist_ok=True)
        app.mount("/frames", StaticFiles(directory=str(frames_path)), name="frames")
        mounted["frames"] = True

    return resolved_video_path, frames_path


# Helper function for SSE-enhanced /extract_frames_and_embeddings endpoint
def event_stream() -> Generator[str, None, None]:
    global image_paths, image_embeddings

    yield "data: Starting frame extraction and embedding computation...\n\n"

    for video_file in os.listdir(app.state.VIDEO_DIR):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(app.state.VIDEO_DIR, video_file)
            msg = f"Extracting frames from {video_path}"
            logger.info(msg)
            yield f"data: {msg}\n\n"

            for line in extract_frames(video_path, app.state.VIDEO_DIR):
                yield f"data: {line}\n\n"
        else:
            msg = f"Skipping non-video file: {video_file}"
            logger.info(msg)
            yield f"data: {msg}\n\n"

    yield "data: Extracting image embeddings...\n\n"

    (
        image_paths,
        image_embeddings,
    ) = get_image_embeddings(app.state.FRAMES_DIR, device=device, clip_model=clip_model)

    msg = f"Processed {len(image_paths)} images."
    logger.info(msg)
    yield f"data: {msg}\n\n"

    _, m = build_hnsw_index(image_embeddings, image_paths, db_dir=DB_DIR)
    msg = f"Built hnsw index and metadata file with {len(m)} entries!"
    logger.info(msg)
    yield f"data: {msg}\n\n"
    yield "data: DONE\n\n"


@app.post("/extract_frames_and_embeddings")
def extract_frames_and_compute_embeddings(req: AnalysisRequest):
    if not app.state.VIDEO_DIR:
        raise HTTPException(
            status_code=400, detail="Пожалуйста, сначала укажите рабочую папку с видео!"
        )
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# TODO - move somewhere
mounted = {"videos": False, "frames": False}


@app.post("/mount_and_list")
def mount_and_list_videos(data: FolderRequest):
    folder_path = Path(data.path).resolve()
    folder_path, frames_path = mount_folders(folder_path)

    if not folder_path.is_dir():
        raise HTTPException(
            status_code=400, detail="Provided path is not a valid directory"
        )

    app.state.VIDEO_DIR = folder_path
    app.state.FRAMES_DIR = frames_path

    # Collect video file full paths
    video_exts = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
    video_files = [
        f"{SERVER_VIDEO_DIR}{f.name}"
        for f in folder_path.iterdir()
        if f.is_file() and f.suffix in video_exts
    ]

    return JSONResponse(
        {
            "mounted_video_dir": str(folder_path),
            "mounted_frames_dir": str(frames_path) if frames_path.is_dir() else None,
            "video_files": video_files,
        }
    )


# TODO - update "no embeddings" warning after integrating vector db
@app.post("/query_similar_images")
def query_similar_images(req: QueryRequest):
    mount_folders(app.state.VIDEO_DIR)

    hnsw_index, metadata = load_hnsw_index(db_dir=DB_DIR)
    if hnsw_index is None or metadata is None:
        return JSONResponse(content={"error": "Images not loaded yet"}, status_code=400)

    # Translate query to English with translation model
    translated_query = ru_to_en(
        req.query, device=device, tr_model=tr_model, tokenizer=tokenizer
    ).lower()
    logger.info(f"Received query: {req.query} → {translated_query}")

    # Get top images' filenames and scores
    top_paths, top_scores, video_names, timestamps, image_names = retrieve_top_k(
        translated_query,
        hnsw_index,
        metadata,
        device=device,
        clip_model=clip_model,
        processor=processor,
        k=req.k,
    )
    return {
        "results": [
            {
                "image_path": str(p),
                "score": float(s),
                "video_name": str(vn),
                "timestamp": str(ts),
                "image_name": str(imn),
            }
            for p, s, vn, ts, imn in zip(
                top_paths, top_scores, video_names, timestamps, image_names
            )
        ]
    }


@app.get("/generate_scene_captions")
def generate_scene_captions():
    logger.info("Loading BLIP2 model...")

    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(device)

    logger.info("Getting captions...")
    hnsw_index, metadata = load_hnsw_index(db_dir=DB_DIR)
    if hnsw_index is None or metadata is None:
        return JSONResponse(content={"error": "Images not loaded yet"}, status_code=400)

    # Load image embeddings from the hnsw index
    try:
        num_elements = len(metadata)
        dim = hnsw_index.dim  # assuming hnswlib index was loaded earlier
        image_embeddings = np.zeros((num_elements, dim), dtype=np.float32)
        for i in range(num_elements):
            image_embeddings[i] = hnsw_index.get_items([i])[0]
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to load embeddings: {str(e)}"}, status_code=500
        )

    # Step 1: Cluster the embeddings
    logger.info("Running clustering...")
    cluster_labels = cluster_embeddings(image_embeddings)

    # Step 2: Group metadata by cluster ID
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)  # Save index instead of path

    # Step 3: Get centroid image per cluster
    logger.info("Getting centroid images...")
    centroid_images = get_centroid_images(clusters, image_embeddings)

    # Step 4: Generate captions using BLIP2
    logger.info("Generating captions...")
    build_csv = build_captions_csv(
        centroid_images,
        metadata,
        device,
        blip_model,
        blip_processor,
        csv_output_dir=REPORT_DIR,
    )
    if build_csv:
        logger.info(f"Captions saved to {REPORT_DIR}/scene_captions.csv.")
    else:
        logger.warning("Captions failed to build!")

    logger.info("Deleting BLIP2 models and cleaning up...")
    # Unload the model and processor
    del blip_model
    del blip_processor

    # Run garbage collection
    gc.collect()

    # If using CUDA, clear the VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# TODO - timeout after 30/40s and ask user to try again (with a toast?)
@app.post("/generate_pdf_report")
def generate_pdf_report(req: OpenrouterRequest):
    api_key = f"Bearer {req.openrouter_api_key}"
    model = req.openrouter_model

    csv_input_path = Path(f"{REPORT_DIR}/scene_captions.csv")
    if not csv_input_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Файл не найден: {csv_input_path}. Возможно, вы еще не запускали /generate_scene_captions?"
            },
        )

    # Open CSV with BLIP2 captions
    centroid_captions = read_captions_csv(csv_input_path)

    # Query LLM with structured input
    logger.info("Requesting LLM scenarios...")
    llm_response = request_scenarios(
        centroid_captions, api_key=api_key, model=model, response_log_dir=REPORT_DIR
    )
    raw_content = llm_response.json()["choices"][0]["message"]["content"]
    llm_data = decode_response(raw_content)

    # Create PDF
    # TODO - name report with used model + timestamp?
    if not os.path.exists(REPORT_DIR):
        logger.info(f"Reports directory not found, creating {REPORT_DIR}.")
        os.makedirs(REPORT_DIR)
    files_dir = REPORT_DIR + "/lib"
    output_pdf = REPORT_DIR + "/report.pdf"
    ready_pdf = prepare_pdf(llm_data, files_dir, output_pdf)
    ready_pdf.save()
    logger.info(f"PDF created at {output_pdf}")

    # Return PDF to user
    return FileResponse(
        output_pdf, media_type="application/pdf", filename="scenario_report.pdf"
    )


@app.post("/run_whisper")
def analyze_with_whisper(req: WhisperRequest):
    segments = []
    for audio_file in os.listdir(app.state.VIDEO_DIR):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(app.state.VIDEO_DIR, audio_file)
            file_segments = run_whisper(
                audio_path, model=req.model, batch_size=req.batch_size
            )
            segments.extend(file_segments)
        else:
            msg = f"Skipping non-audio file: {audio_file}"
            logger.info(msg)

    out_path = os.path.join(app.state.VIDEO_DIR, "whisper_transcripts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return segments


@app.get("/build_whisper_db")
def build_whisper_db():
    global whisper_store
    transcript_path = Path(app.state.VIDEO_DIR) / "whisper_transcripts.json"
    if not transcript_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Файл транскрипции не найден! Сначала запустите /run_whisper",
        )

    segments = load_segments(transcript_path)
    documents = build_documents(segments)
    whisper_store = embed_and_save_index(documents, DB_DIR)


@app.post("/search_whisper_db")
def search_whisper_db(req: WhisperSearchRequest):
    global whisper_store
    if not whisper_store:
        whisper_store = load_index(DB_DIR)
    return semantic_search(req.query, whisper_store)


@app.post("/search_whisper_db_by_topic")
def search_whisper_by_topic(req: WhisperTopicRequest):
    global whisper_store
    if not whisper_store:
        whisper_store = load_index(DB_DIR)
    return search_by_topic(req.topic, whisper_store)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
