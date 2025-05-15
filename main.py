import torch
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Generator
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from peft import PeftModel

from helpers import (
    get_image_embeddings,
    retrieve_top_k,
    extract_frames,
    ru_to_en,
    cluster_embeddings,
    group_by_cluster,
    get_centroid_images,
    describe_image,
    request_scenarios,
    decode_response,
    prepare_pdf,
)

# TODO - make the app clean up the frames folder (maybe something else?) when exiting

# Logging ——————————————————————————————————————————————
log_file = "app.log"
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

# TODO - make the user provide their own api key
OPENROUTER_API_KEY = (
    "Bearer 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"
)
# TODO - make the model changeable
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-30b-a3b:free"

# TODO - allow user to set their own input video dir
video_directory = "./videos"
output_directory = f"{video_directory}/frames"
# TODO - name report with timestamp / used LLM model etc
output_pdf = "./reports/report.pdf"

clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").to(
    device
)
clip_model = PeftModel.from_pretrained(clip_model, "./models/LoRA_wedding")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
tr_model = (
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    .to(device)
    .eval()
)

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
).to(device)

# GLOBAL STATE
image_paths, image_embeddings = [], None

# FastAPI App ———————————————————————————————————————————
app = FastAPI()


# Query class to feed into /query_similar_images
class QueryRequest(BaseModel):
    query: str
    k: int = 3


# Helper function for SSE-enhanced /extract_frames_and_embeddings endpoint
# TODO - extend with support for hnsw index
# TODO - update with ffmpeg output -> SSE stream
def event_stream() -> Generator[str, None, None]:
    global image_paths, image_embeddings

    yield "data: Starting frame extraction and embedding computation...\n\n"

    for video_file in os.listdir(video_directory):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_directory, video_file)
            msg = f"Extracting frames from {video_path}"
            logger.info(msg)
            yield f"data: {msg}\n\n"

            extract_frames(video_path, output_directory)
        else:
            warn = f"Skipping non-video file: {video_file}"
            logger.warning(warn)
            yield f"data: {warn}\n\n"

    yield "data: Extracting image embeddings...\n\n"

    image_paths, image_embeddings = get_image_embeddings(
        output_directory, device=device, clip_model=clip_model
    )

    msg = f"Processed {len(image_paths)} images."
    logger.info(msg)
    yield f"data: {msg}\n\n"

    yield "data: DONE\n\n"


@app.get("/extract_frames_and_embeddings")
def extract_frames_and_compute_embeddings():
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# TODO - update "no embeddings" warning after integrating vector db
@app.post("/query_similar_images")
def query_similar_images(req: QueryRequest):
    if not image_paths or image_embeddings is None:
        return JSONResponse(content={"error": "Images not loaded yet"}, status_code=400)

    # Translate query to English with translation model
    translated_query = ru_to_en(
        req.query, device=device, tr_model=tr_model, tokenizer=tokenizer
    ).lower()
    logger.info(f"Received query: {req.query} → {translated_query}")

    # TODO - add timestamps to results
    # Get top images' filenames and scores
    top_paths, scores = retrieve_top_k(
        translated_query,
        image_paths,
        image_embeddings,
        device=device,
        clip_model=clip_model,
        processor=processor,
        k=req.k,
    )
    return {
        "results": [
            {"filename": os.path.basename(p), "score": float(s)}
            for p, s in zip(top_paths, scores)
        ]
    }


@app.get("/generate_pdf_report")
def generate_pdf_report():
    global image_paths, image_embeddings
    if not image_paths or image_embeddings is None:
        return JSONResponse(content={"error": "Images not loaded yet"}, status_code=400)

    # Cluster image embeddings to get separate 'scenes'
    logger.info("Running clustering...")
    cluster_labels = cluster_embeddings(image_embeddings)
    clusters = group_by_cluster(image_paths, cluster_labels, include_noise=True)

    # Get scenes' centroids
    logger.info("Getting centroid images...")
    centroid_images = get_centroid_images(clusters, image_embeddings, image_paths)

    # Run BLIP2 on scenes' centroids to get their captions
    logger.info("Generating captions...")
    centroid_captions = []
    for cluster_id, img_path in tqdm(centroid_images.items()):
        caption = describe_image(
            img_path,
            device=device,
            blip_model=blip_model,
            blip_processor=blip_processor,
        )
        centroid_captions.append(
            {
                "cluster_id": cluster_id,
                "image_path": str(img_path),
                "caption": caption,
            }
        )

    # Query LLM for a scenario based on the scenes' captions
    logger.info("Requesting LLM scenarios...")
    llm_response = request_scenarios(
        centroid_captions, api_key=OPENROUTER_API_KEY, model=DEFAULT_OPENROUTER_MODEL
    )
    raw_content = llm_response.json()["choices"][0]["message"]["content"]
    llm_data = decode_response(raw_content)

    # Prepare PDF and save it
    logger.info("Creating PDF...")
    ready_pdf = prepare_pdf(llm_data, output_pdf)
    ready_pdf.save()
    logger.info(f"PDF created at {output_pdf}")

    # Send PDF to user
    return FileResponse(
        output_pdf, media_type="application/pdf", filename="scenario_report.pdf"
    )
