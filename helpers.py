import os
import subprocess
import requests
import json
import numpy as np
from collections import defaultdict
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
from torchvision import transforms

import hdbscan
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


# Find top-3 most similar images
def retrieve_top_k(
    query, image_paths, image_embeddings, device, clip_model, processor, k=3
):
    text_emb = get_text_embedding(query, device, clip_model, processor)
    similarities = (image_embeddings @ text_emb.T).squeeze()
    top_k_indices = similarities.topk(k).indices
    return [image_paths[i] for i in top_k_indices], similarities[top_k_indices].numpy()


# FFMPEG FUNCTIONS ———————————————————————
# TODO - downscale the video before grabbing frames
def extract_frames(video_path, output_dir, fps=1 / 3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.basename(video_path).split(".")[0]
    output_template = os.path.join(output_dir, f"{video_name}+%d.jpg")
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"scale=480:-2,fps={fps}",
        output_template,
    ]
    subprocess.run(command)


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
    embeddings_np = embeddings.numpy()
    embeddings_np = normalize(embeddings_np)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_np)

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
def get_centroid_images(clusters, image_embeddings, image_paths):
    centroid_images = {}
    embeddings_np = image_embeddings.numpy()

    for cluster_id, img_paths in clusters.items():
        indices = [image_paths.index(p) for p in img_paths]
        cluster_embeddings = embeddings_np[indices]

        centroid = cluster_embeddings.mean(axis=0)
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(dists)

        centroid_images[cluster_id] = img_paths[closest_idx]
    return centroid_images


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
        {"image_path": item["image_path"], "description": item["caption"].strip("\n")}
        for item in centroid_captions
    ]

    if not prompt:
        prompt = f"""
        Ты опытный видеомонтажёр, специализирующийся на свадебных видео.  
        Тебе нужно составить логическую последовательность из следующих сцен (под ключами description):

        {json.dumps(scenes, indent=2, ensure_ascii=False)}

        Сделай следующее:

        1. 🔢 Расставь сцены в том порядке, который создаёт плавную и эмоционально выразительную историю.
        2. 💡 Для каждой сцены напиши краткое обоснование: почему она должна идти именно здесь?
        3. После каждой сцены укажи ее image_path без дополнительных символов.
        4. 🎬 В конце объясни общую логику выбранного сценария и его эмоциональной дуги.

        Структурируй ответ в формате JSON в следующей структуре:
        {{
        "scenes": [
            {{
            "image_path": "путь_к_изображению",
            "description": "описание сцены",
            "justification": "обоснование"
            }},
        ],
        "overall_logic": "логика сценария"
        }}
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
        image_path = scene["image_path"]
        description = scene["description"]
        justification = scene["justification"]
        video_filename, pseudo_timestamp = os.path.splitext(
            os.path.basename(image_path)
        )[0].split("+")

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
