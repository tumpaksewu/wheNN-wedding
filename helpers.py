import os
import subprocess
import requests
import json
import numpy as np
import pickle
import csv
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

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
def build_hnsw_index(embeddings_tensor, image_paths, db_dir):
    index_path = f"{db_dir}/hnsw_index.bin"
    metadata_path = f"{db_dir}/hnsw_metadata.pkl"

    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

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
    image_names = []
    for posix_path in image_paths:
        filename = posix_path.name
        image_name = os.path.basename(filename)
        video_name, frame_number_str = filename.split("~")
        video_name = os.path.basename(video_name)
        frame_number = int(os.path.splitext(frame_number_str)[0])
        timestamp = frame_number * 3  # 3 sec/frame assumption
        video_filenames.append(video_name)
        timestamps.append(timestamp)
        image_names.append(image_name)

    # Save metadata
    metadata = {
        i: [str(image_paths[i]), video_filenames[i], timestamps[i], image_names[i]]
        for i in range(len(image_paths))
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index and metadata saved: {index_path}, {metadata_path}")
    return p, metadata


# Load hnswlib index + metadata
def load_hnsw_index(db_dir, dim=512):
    index_path = os.path.abspath(os.path.join(db_dir, "hnsw_index.bin"))
    metadata_path = os.path.abspath(os.path.join(db_dir, "hnsw_metadata.pkl"))

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
    image_names = [metadata[i][3] for i in labels[0]]
    top_scores = [1 - d for d in distances[0]]

    return top_paths, top_scores, video_filenames, timestamps, image_names


# FFMPEG FUNCTIONS ———————————————————————
def extract_frames(
    video_path: str, output_dir: str, fps: float = 1 / 3
) -> Generator[str, None, None]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_name = os.path.basename(video_path)
    output_template = os.path.join(output_dir, f"{video_name}~%d.jpg")

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
    embeddings, n_components=32, min_cluster_size_ratio=0.005, min_samples=3
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
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=0.05,
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


def build_captions_csv(
    centroid_images, metadata, device, blip_model, blip_processor, csv_output_dir
):
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_output_path = f"{csv_output_dir}/scene_captions.csv"
    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "cluster_id",
            "image_path",
            "video_filename",
            "timestamp",
            "caption",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cluster_id, img_idx in tqdm(centroid_images.items()):
            img_path, video_filename, timestamp, _ = metadata[img_idx]
            caption = describe_image(
                img_path,
                device=device,
                blip_model=blip_model,
                blip_processor=blip_processor,
            )
            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "image_path": str(img_path),
                    "video_filename": video_filename,
                    "timestamp": timestamp,
                    "caption": caption,
                }
            )
    return True


# TODO - provide error if csv_input_path doesn't exist
def read_captions_csv(csv_input_path):
    centroid_captions = []
    with open(csv_input_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            centroid_captions.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "image_path": row["image_path"],
                    "video_filename": row["video_filename"],
                    "timestamp": int(row["timestamp"]),
                    "caption": row["caption"],
                }
            )
    return centroid_captions


# LLM FUNCTIONS ———————————————————————
def request_scenarios(
    centroid_captions, api_key, model, prompt=None, response_log_dir=None
):
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
            "timestamp": "временной_код"
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

    if response_log_dir:
        os.makedirs(response_log_dir, exist_ok=True)
        try:
            response_json = response.json()
        except Exception as e:
            response_json = {
                "error": f"Failed to decode JSON: {str(e)}",
                "raw_text": response.text,
            }
        with open(f"{response_log_dir}/llm_response.json", "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

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
def safe_text(text):
    """Обеспечивает корректное отображение текста"""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8")
    return str(text)


def format_timestamp(seconds):
    """Форматирует время из секунд в чч:мм:сс"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def draw_wrapped_text(
    c,
    text,
    x,
    y,
    max_width,
    font_name="DejaVu",
    font_size=12,
    leading=14,
    color=colors.black,
):
    """Рисует текст с переносом строк"""
    text = safe_text(text)
    text_object = c.beginText(x, y)
    text_object.setFont(font_name, font_size)
    text_object.setLeading(leading)
    text_object.setFillColor(color)

    words = text.split()
    line = []
    line_count = 0

    for word in words:
        test_line = " ".join(line + [word]) + " "
        if c.stringWidth(test_line, font_name, font_size) <= max_width:
            line.append(word)
        else:
            text_object.textLine(" ".join(line))
            line = [word]
            line_count += 1
    if line:
        text_object.textLine(" ".join(line))
        line_count += 1

    c.drawText(text_object)
    return y - (line_count * leading)


def add_scene_table(c, scene_data, x, y, width, height, font_name="DejaVu"):
    """Создаёт компактную таблицу для сцены с прозрачным фоном"""
    description = safe_text(scene_data["description"])
    justification = safe_text(scene_data["justification"])
    file_info = safe_text(scene_data["video_filename"])
    time_formatted = format_timestamp(scene_data["timestamp"])

    # Разбиваем обоснование на строки
    justification_lines = []
    current_line = []
    max_line_width = width * 0.65
    font_size = 8  # Уменьшенный размер шрифта для таблицы

    for word in justification.split():
        test_line = " ".join(current_line + [word])
        if c.stringWidth(test_line, font_name, font_size) <= max_line_width:
            current_line.append(word)
        else:
            justification_lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        justification_lines.append(" ".join(current_line))

    data = [["Описание", description]]
    data.append(["Обоснование", justification_lines[0]])
    for line in justification_lines[1:]:
        data.append(["", line])
    data.extend([["Файл", file_info], ["Таймкод", time_formatted]])

    table = Table(data, colWidths=[width * 0.3, width * 0.7], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BACKGROUND", (0, 1), (-1, -1), None),  # Прозрачный фон
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    if len(justification_lines) > 1:
        table.setStyle(TableStyle([("SPAN", (0, 1), (0, len(justification_lines)))]))
    table.wrapOn(c, width, height)
    table.drawOn(c, x, y - table._height)
    return y - table._height - 0.3 * cm


def draw_background(c, width, height, bg_path):
    """Рисует фоновое изображение на всю страницу"""
    try:
        bg = ImageReader(bg_path)
        c.drawImage(
            bg,
            0,
            0,
            width=width,
            height=height,
            preserveAspectRatio=False,
            mask="auto",
        )
    except Exception as e:
        print(f"⚠️ Ошибка загрузки фона: {str(e)}")


def draw_logo(c, width, margin, y, logo_path):
    """Добавляет логотип в верхнюю часть документа"""
    try:
        logo = ImageReader(logo_path)
        logo_width = 4 * cm  # Средний размер логотипа
        logo_height = logo_width * 0.5  # Сохраняем пропорции

        # Позиционируем логотип по центру
        x = (width - logo_width) / 2
        c.drawImage(
            logo,
            x,
            y - logo_height,
            width=logo_width,
            height=logo_height,
            mask="auto",
        )
        return logo_height + 0.5 * cm  # Возвращаем высоту логотипа + отступ
    except Exception as e:
        print(f"⚠️ Ошибка загрузки логотипа: {str(e)}")
        return 0
    return 0


def prepare_pdf(content_data, files_dir, output_pdf="wedding_scenario_reportlab.pdf"):
    # Настройка шрифтов
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", f"{files_dir}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(
            TTFont("DejaVu-Bold", f"{files_dir}/DejaVuSans-Bold.ttf")
        )
        main_font = "DejaVu"
        bold_font = "DejaVu-Bold"
    except Exception:
        pdfmetrics.registerFont(TTFont("ArialUnicode", "arial.ttf"))
        main_font = bold_font = "ArialUnicode"

    # Создание PDF
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 1.5 * cm
    bottom_margin = 2 * cm  # Увеличенный нижний отступ
    y = height - margin
    max_width = width - 2 * margin

    # Фон на всю страницу
    bg_path = f"{files_dir}/background.jpeg"
    draw_background(c, width, height, bg_path)

    # === Первая страница ===
    # Добавляем логотип
    logo_file = f"{files_dir}/logo.jpg"
    logo_height = draw_logo(c, width, margin, y, logo_file)
    y -= logo_height + 0.5 * cm  # Отступ после логотипа

    # Стили текста
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=10,
    )

    # Блок: Общая логика сценария
    title = Paragraph("Сценарий", title_style)
    title.wrapOn(c, max_width, height)
    title.drawOn(c, margin, y)
    y -= title.height + 0.3 * cm

    logic_style = ParagraphStyle(
        "LogicStyle",
        parent=styles["BodyText"],
        fontName=main_font,
        fontSize=10,
        textColor=colors.HexColor("#34495e"),
        leading=14,
        spaceBefore=5,
        spaceAfter=10,
    )
    logic_text = Paragraph(safe_text(content_data["overall_logic"]), logic_style)
    logic_text.wrapOn(c, max_width, height)
    logic_text.drawOn(c, margin, y - logic_text.height)
    y -= logic_text.height + 1 * cm

    # Разделитель
    c.setStrokeColor(colors.HexColor("#7f8c8d"))
    c.setLineWidth(1)
    c.line(margin, y, width - margin, y)
    y -= 1 * cm

    # Заголовок "Сцены свадебного видео" только на первой странице
    scenes_title = Paragraph(
        "Рекомендованный порядок сцен для монтажа видео", title_style
    )
    scenes_title.wrapOn(c, max_width, height)
    scenes_title.drawOn(c, margin, y)
    y -= scenes_title.height + 0.5 * cm

    # === Обработка сцен ===
    scenes_per_page = [2]  # На первой странице 2 сцены
    scenes_per_page.extend(
        [3] * ((len(content_data["scenes"]) - 2 + 2) // 3)
    )  # На остальных по 3

    current_page = 0
    scenes_on_current_page = 0

    for i, scene in enumerate(content_data["scenes"], 1):
        # Проверяем, нужно ли начинать новую страницу
        if scenes_on_current_page >= scenes_per_page[current_page]:
            c.showPage()
            current_page += 1
            scenes_on_current_page = 0
            y = height - margin
            draw_background(c, width, height, bg_path)

        # Заголовок сцены
        c.setFont(bold_font, 12)
        c.setFillColor(colors.HexColor("#16a085"))
        y = draw_wrapped_text(
            c,
            f"Сцена #{i}",
            margin,
            y,
            max_width=max_width,
            font_name=bold_font,
            font_size=12,
            leading=14,
            color=colors.HexColor("#16a085"),
        )
        y -= 0.3 * cm

        # Вставка изображения
        if os.path.isfile(scene["image_path"]):
            try:
                img = ImageReader(scene["image_path"])
                iw, ih = img.getSize()
                aspect = ih / iw
                img_width = min(max_width, 5 * cm)  # Фиксированная ширина изображения
                img_height = img_width * aspect

                # Ограничиваем максимальную высоту изображения
                max_img_height = 3 * cm
                if img_height > max_img_height:
                    img_height = max_img_height
                    img_width = img_height / aspect

                c.drawImage(
                    img,
                    margin,
                    y - img_height,
                    width=img_width,
                    height=img_height,
                    mask="auto",
                )

                # Подпись к изображению
                c.setFont(main_font, 7)
                c.setFillColor(colors.HexColor("#7f8c8d"))
                c.drawString(
                    margin,
                    y - img_height - 0.4 * cm,
                    f"Кадр из видео: {safe_text(scene['video_filename'])} (time code:{format_timestamp(scene['timestamp'])})",
                )
                y -= img_height + 0.6 * cm
            except Exception as e:
                y = draw_wrapped_text(
                    c,
                    f"[Ошибка загрузки изображения: {str(e)}]",
                    margin,
                    y - 10,
                    max_width=max_width,
                    color=colors.red,
                )
        else:
            y = draw_wrapped_text(
                c,
                "[⚠️ Изображение не найдено]",
                margin,
                y - 10,
                max_width=max_width,
                color=colors.red,
            )

        # Таблица с информацией о сцене
        y = add_scene_table(c, scene, margin, y, max_width, height, main_font)

        # Отступ между сценами
        if scenes_on_current_page < scenes_per_page[current_page] - 1:
            c.setStrokeColor(colors.HexColor("#7f8c8d"))
            c.setLineWidth(1)
            c.line(margin, y, width - margin, y)
            y -= 0.5 * cm

        scenes_on_current_page += 1

    # Подпись внизу последней страницы
    c.setFont(main_font, 7)
    c.setFillColor(colors.HexColor("#7f8c8d"))
    c.drawRightString(width - margin, bottom_margin - 0.5 * cm, "• Generated by wheNN")

    return c
