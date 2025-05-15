import os
import subprocess
import requests
import json
import numpy as np
import pickle
from collections import defaultdict
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Generator

import torch
from torchvision import transforms

import hdbscan
import hnswlib
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
import textwrap


# CLIP FUNCTIONS ———————————————————————
# TODO - evaluate and revert if needed
preprocess = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)
# preprocess = transforms.Compose(
#     [
#         transforms.Resize(
#             (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
#     ]
# )


# Function to load and preprocess image
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return preprocess(image)


# Load images from folder and encode them
def get_image_embeddings(folder_path, device, clip_model):
    folder = Path(folder_path)
    image_paths = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )

    all_embeddings = []
    for img_path in tqdm(image_paths):
        image_tensor = load_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            all_embeddings.append(embedding.cpu())

    return image_paths, torch.cat(all_embeddings, dim=0)


# Encode text query
def get_text_embedding(text, device, clip_model, processor):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


# VECTOR-DB FUNCTIONS ———————————————————————
# Build hnswlib index + store metadata
# TODO - clean up path logic
def build_hnsw_index(
    embeddings_tensor,
    image_paths,
    index_path="./db/hnsw_index.bin",
    metadata_path="./db/hnsw_metadata.pkl",
):
    if not os.path.exists("./db"):
        os.makedirs("./db")

    # Convert to numpy array
    embeddings = embeddings_tensor.numpy().astype(np.float32)
    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]

    # Initialize Hnswlib index
    p = hnswlib.Index(space="cosine", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Add embeddings
    p.add_items(embeddings, ids=list(range(num_elements)))

    # Set ef for runtime search
    p.set_ef(50)

    # Save index
    p.save_index(index_path)

    # Format metadata
    video_filenames = []
    timestamps = []
    for posix_path in image_paths:
        filename = posix_path.name
        video_name, frame_number_str = filename.split("+")
        video_name = os.path.basename(video_name)
        frame_number = int(os.path.splitext(frame_number_str)[0])
        timestamp = frame_number * 3  # 3 sec/frame assumption
        video_filenames.append(video_name)
        timestamps.append(timestamp)

    # Save metadata
    metadata = {
        i: [str(image_paths[i]), video_filenames[i], timestamps[i]]
        for i in range(len(image_paths))
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index and metadata saved: {index_path}, {metadata_path}")
    return p, metadata


# Load hnswlib index + metadata
def load_hnsw_index(
    index_path="./db/hnsw_index.bin", metadata_path="./db/hnsw_metadata.pkl", dim=512
):
    if not os.path.isfile(index_path):
        return None, None
    p = hnswlib.Index(space="cosine", dim=dim)
    p.load_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return p, metadata


# Retrieve top-K using prebuilt hnswlib index + metadata file
def retrieve_top_k(query, hnsw_index, metadata, device, clip_model, processor, k=3):
    text_emb = get_text_embedding(query, device, clip_model, processor)
    text_emb_np = text_emb.numpy().astype("float32")

    labels, distances = hnsw_index.knn_query(text_emb_np, k=k)
    top_paths = [metadata[i][0] for i in labels[0]]
    video_filenames = [metadata[i][1] for i in labels[0]]
    timestamps = [metadata[i][2] for i in labels[0]]
    top_scores = [1 - d for d in distances[0]]

    return top_paths, top_scores, video_filenames, timestamps


# FFMPEG FUNCTIONS ———————————————————————
def extract_frames(
    video_path: str, output_dir: str, fps: float = 1 / 3
) -> Generator[str, None, None]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_name = os.path.basename(video_path)
    output_template = os.path.join(output_dir, f"{video_name}+%d.jpg")

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-ss",
        "3",
        "-vf",
        f"scale=480:-2,fps={fps}",
        output_template,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    for line in process.stdout:
        yield f"[ffmpeg] {line.strip()}"

    process.wait()


# TRANSLATION FUNCTIONS ———————————————————————
def ru_to_en(text: str, device, tr_model, tokenizer) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Tokenizing
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=False,  # Отключаем тренкацию — чтобы не обрезать длинный текст
    ).to(device)
    # Translation generation
    translated_tokens = tr_model.generate(
        **inputs,
        max_length=512,  # Увеличиваем длину вывода
        num_beams=8,  # Больше бимов для лучшего поиска
        length_penalty=1.0,  # Более нейтральное отношение к длине
        early_stopping=False,  # Не останавливаемся преждевременно
        no_repeat_ngram_size=2,  # Избегаем повторений
    )
    # Decoding
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


# CLUSTERING FUNCTIONS ———————————————————————
def cluster_embeddings(
    embeddings, n_components=64, min_cluster_size_ratio=0.015, min_samples=1
):
    # Step 1: Convert & reduce
    embeddings = normalize(embeddings)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    # Step 2: Use dynamic min_cluster_size based on % of video length
    min_cluster_size = max(
        2, int(len(reduced) * min_cluster_size_ratio)
    )  # e.g., 1% of total frames

    # Step 3: Cluster with cosine metric
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean"
    )  # works on PCA-reduced vectors
    cluster_labels = clusterer.fit_predict(reduced)

    return cluster_labels


# Image grouping by cluster
def group_by_cluster(image_paths, cluster_labels, include_noise=True):
    clusters = defaultdict(list)
    for img_path, label in zip(image_paths, cluster_labels):
        if include_noise or label != -1:
            clusters[label].append(img_path)
    return clusters


# Get centroid images for each cluster
def get_centroid_images(clusters, embeddings):
    centroids = {}
    for cluster_id, indices in clusters.items():
        cluster_embeds = embeddings[indices]
        centroid = np.mean(cluster_embeds, axis=0)
        distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
        min_idx = indices[np.argmin(distances)]
        centroids[cluster_id] = min_idx
    return centroids


# Run BLIP2 on centroid images to get captions
def describe_image(image_path, device, blip_model, blip_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


# LLM FUNCTIONS ———————————————————————
def request_scenarios(centroid_captions, api_key, model, prompt=None):
    scenes = [
        {
            "description": item["caption"].strip("\n"),
            "image_path": item["image_path"],
            "video_filename": item["video_filename"],
            "timestamp": item["timestamp"],
        }
        for item in centroid_captions
    ]

    if not prompt:
        prompt = f"""
        Ты опытный видеомонтажёр, специализирующийся на свадебных видео.  
        Тебе нужно составить логическую последовательность из следующих сцен (под ключами description):

        {json.dumps(scenes, indent=2, ensure_ascii=False)}

        Сделай следующее:

        1. Расставь сцены в том порядке, который создаёт плавную и эмоционально выразительную историю.
        2. Для каждой сцены напиши краткое обоснование: почему она должна идти именно здесь.
        3. В конце объясни общую логику выбранного сценария и его эмоциональной дуги.
        4. Также, не забудь отправить обратно информацию о изображении и видеофайле (image_path, video_filename, timestamp). Придерживайся структуры ниже.

        Отправь ответ четко в формате валидного JSON в следующей структуре:
        {{
        "scenes": [
            {{
            "description": "описание сцены",
            "justification": "обоснование",
            "image_path": "путь_к_изображению",
            "video_filename": "название_видеофайла",
            "timestamp": "временной_код",
            }},
        ],
        "overall_logic": "Общая логика сценария"
        }}
        Кроме этого JSON в твоем ответе не должно быть ничего.
        Формируй ответ понятно, без использования служебных слов типа "currentIndex", "kola!" и других технических терминов.
        """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )
    return response


def decode_response(raw_content):
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract the JSON manually in case there's surrounding text
        import re

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in model response")
    return data


# PDF FUNCTIONS ———————————————————————
def draw_wrapped_text(c, text, x, y, max_width, font_size=12, leading=14):
    c.setFont("DejaVu", font_size)
    lines = textwrap.wrap(text, width=int(max_width / (font_size * 0.45)))
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


# TODO - make a better pdf
def prepare_pdf(parsed_data, output_pdf="wedding_scenario_reportlab.pdf"):
    # === Register font with Cyrillic support ===
    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSansCondensed.ttf"))

    # === Setup PDF ===
    c = canvas.Canvas(output_pdf, pagesize=A4)
    c.setFont("DejaVu", 14)
    width, height = A4
    margin = 2 * cm
    y = height - margin
    max_width = width - 2 * margin

    # === Title ===
    c.setFont("DejaVu", 18)
    c.drawCentredString(width / 2, y, "Предложенный порядок сцен и обоснование")
    y -= 2 * cm

    # === Draw scenes ===
    for i, scene in enumerate(parsed_data["scenes"], 1):
        description = scene["description"]
        justification = scene["justification"]
        image_path = scene["image_path"]
        video_filename = scene["video_filename"]
        timestamp = scene["timestamp"]

        c.setFont("DejaVu", 14)
        y = draw_wrapped_text(
            c,
            f"Сцена {i}: {description}",
            margin,
            y,
            max_width=max_width,
            font_size=13,
            leading=15,
        )
        y = draw_wrapped_text(
            c,
            f"Обоснование: {justification}",
            margin,
            y - 5,
            max_width=max_width,
            font_size=11,
            leading=13,
        )
        y = draw_wrapped_text(
            c,
            f"Данные файла: {image_path}, {video_filename}, {timestamp}",
            margin,
            y - 5,
            max_width=max_width,
            font_size=11,
            leading=13,
        )

        # Insert image if available
        if os.path.isfile(image_path):
            try:
                img = ImageReader(image_path)
                iw, ih = img.getSize()
                aspect = ih / iw
                img_width = (width - 2 * margin) / 2
                img_height = (img_width * aspect) / 2
                if y - img_height < margin:
                    c.showPage()
                    c.setFont("DejaVu", 14)
                    y = height - margin
                c.drawImage(
                    img, margin, y - img_height, width=img_width, height=img_height
                )
                y -= img_height + 1 * cm
            except Exception as e:
                y = draw_wrapped_text(
                    c,
                    f"[Image failed to load: {e}]",
                    margin,
                    y - 10,
                    max_width=max_width,
                )
        else:
            y = draw_wrapped_text(
                c, "[⚠️ Изображение не найдено]", margin, y - 10, max_width=max_width
            )

        if y < 5 * cm:
            c.showPage()
            c.setFont("DejaVu", 14)
            y = height - margin

    # === Draw overall logic at the end ===
    c.showPage()
    y = height - margin
    c.setFont("DejaVu", 16)
    c.drawCentredString(width / 2, y, "🎬 Общая логика сценария")
    y -= 2 * cm
    y = draw_wrapped_text(
        c,
        parsed_data["overall_logic"],
        margin,
        y,
        max_width=max_width,
        font_size=12,
        leading=15,
    )
    return c
