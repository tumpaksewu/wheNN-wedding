from nicegui import ui
import httpx
import json
import socket
import os
import subprocess
import sys
from datetime import timedelta
from typing import Optional
import tkinter as tk
from tkinter import filedialog

# Конфигурация
API_URL = "http://localhost:8000"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}


class AppState:
    def __init__(self):
        self.video_dir: Optional[str] = None
        self.selected_video: Optional[str] = None
        self.openrouter_user_api_key: str = ""
        self.openrouter_user_model: str = ""
        self.openrouter_default_api_key: str = (
            "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"
        )
        self.openrouter_default_model: str = "qwen/qwen3-30b-a3b:free"
        self.query_text: str = ""
        self.query_results = []
        self.progress: float = 0
        self.progress_message: str = ""
        self.show_mount_button = False
        self.show_extract_button = False
        self.show_video_settings = True
        self.k = 6


state = AppState()
ui.dark_mode().enable()

ui.add_head_html("""
<style>
.q-dialog__backdrop {
    background-color: rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(8px);
}
</style>
""")


def select_folder():
    """Функция выбора папки с использованием Tkinter"""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)

    try:
        folder = filedialog.askdirectory(title="Выберите папку с видео")
        if folder:
            state.show_mount_button = True
            state.video_dir = folder
            video_dir_input.value = folder
            ui.notify(f"Выбрана папка: {folder}", type="positive")
    except Exception as e:
        ui.notify(f"Ошибка при выборе папки: {str(e)}", type="negative")


async def mount_and_list():
    if not state.video_dir:
        ui.notify("Пожалуйста, выберите папку с видео", type="negative")
        return

    try:
        response = httpx.post(
            f"{API_URL}/mount_and_list", json={"path": state.video_dir}, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        state.show_extract_button = True

        ui.notify(
            f"Успешно загружено {len(data['video_files'])} видео", type="positive"
        )
    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}", type="negative")


# Create the dialog only once
progress_dialog = ui.dialog()
with progress_dialog:
    with ui.card().classes("w-96"):
        ui.label("Обработка видео").classes("text-xl font-bold")
        progress_area = ui.log(max_lines=50).classes("h-64")


# TODO - trigger /generate_scene_captions after running this
async def extract_frames_and_embeddings():
    if not state.video_dir:
        ui.notify("Пожалуйста, выберите папку с видео", type="negative")
        return

    state.progress = 0
    progress_area.clear()
    progress_dialog.open()

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/extract_frames_and_embeddings",
                json={"path_to_videos": state.video_dir},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        message = line[5:].strip()

                        if message == "DONE":
                            state.progress = 100
                            progress_area.push("✅ Завершено.")
                            state.show_video_settings = False  # <- Hide the label here
                            ui.notify("Обработка завершена", type="positive")
                            break

                        # Add line to UI
                        progress_area.push(message)

                        # Fake progress bar increment
                        if state.progress < 95:
                            state.progress += 5

        progress_dialog.close()

    except Exception as e:
        progress_area.push(f"❌ Ошибка: {str(e)}")
        ui.notify(f"Ошибка: {str(e)}", type="negative")
        progress_dialog.close()


async def query_similar_images(k: int):
    if not state.query_text:
        ui.notify("Пожалуйста, введите запрос", type="negative")
        return

    try:
        response = httpx.post(
            f"{API_URL}/query_similar_images",
            json={"query": state.query_text, "k": k},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        state.query_results = data.get("results", [])
        update_results_display()

        ui.notify(f"Найдено {len(state.query_results)} результатов", type="positive")
    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}", type="negative")


async def generate_pdf_report():
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{API_URL}/generate_pdf_report",
                json={
                    "openrouter_api_key": state.openrouter_user_api_key
                    or state.openrouter_default_api_key,
                    "openrouter_model": state.openrouter_user_model
                    or state.openrouter_default_model,
                },
            )

        response.raise_for_status()

        pdf_path = "scenario_report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        ui.notify(f"PDF отчет сохранен как {pdf_path}", type="positive")
        open_pdf_button.visible = True

    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}", type="negative")


video_dialog = ui.dialog()
with video_dialog:
    video_element = ui.html(
        '<video id="preview-player" controls class="w-full rounded"></video>'
    )


def show_video_preview(video_name: str, timestamp: float):
    video_url = f"{API_URL}/videos/{video_name}"
    js_code = f"""
        const video = document.getElementById("preview-player");
        video.src = "{video_url}";
        video.load();
        video.onloadedmetadata = () => {{
            video.currentTime = {timestamp};
            video.play();
        }};
    """
    ui.run_javascript(js_code)
    video_dialog.open()


def create_on_click_mrk_button(video_name, timestamp):
    def on_click():
        ui.notify("⏳ Sending to Resolve...", color="warning")

        # FIXME - provide actual path instead of the bullshit on next line
        # FIXME - grab the query name and use it as marker name / note
        # TODO - also make the color selectable
        success = send_payload_to_resolve(
            video_path=f"/Users/aleko/Downloads/test_video_dir/{video_name}",
            target_marker_secs=timestamp,
            marker_color="Lemon",
            marker_name="test",
            marker_note="testttt",
        )

        if success:
            ui.notify("✅ Marker added!", color="positive")
        else:
            ui.notify("❌ Failed to add marker", color="negative")

    return on_click



def update_results_display():
    results_container.clear()
    with results_container:
        with ui.row().classes("gap-4 flex-wrap"):
            for result in state.query_results:
                video_name = result["video_name"]
                timestamp = result["timestamp"]
                image_name = result["image_name"]
                score = result["score"]

                with ui.element("div").classes("cursor-pointer"
                    # TODO - bring back the player
                    # .on(
                    #     "click",
                    #     lambda e, r=result: show_video_preview(
                    #         r["video_name"], r["timestamp"]
                    #     ),
                    ):
                    with (
                        ui.card()
                        .tight()
                        .classes("rounded-lg w-64")
                        .style("box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);")
                    ):
                        ui.image(f"{API_URL}/frames/{image_name}").classes(
                            "w-full rounded-t-lg"
                        )
                        with ui.card_section():
                            ui.label(f"🎞 Видео: {video_name}")
                            ui.label(
                                f"🕑 Время: {str(timedelta(seconds=int(timestamp)))}"
                            )
                            ui.label(f"🔍 Сходство: {score:.2f}")

                            # ✅ Button gets the correct per-card handler
                            ui.button("Send Marker", on_click=create_on_click_mrk_button(video_name, timestamp)).props(
                                "color=accent"
                            )


def open_pdf():
    pdf_path = "scenario_report.pdf"
    if os.path.exists(pdf_path):
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", pdf_path])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", pdf_path])
        elif sys.platform.startswith("win"):
            os.startfile(pdf_path)
        else:
            ui.notify("Неизвестная платформа, не могу открыть PDF", type="negative")
    else:
        ui.notify("PDF файл не найден", type="negative")


def prepare_payload(
    video_path,
    target_marker_secs,
    marker_color,
    marker_name,
    marker_note,
    exit_command=False,
):
    if not exit_command:
        payload = {
            "VIDEO_PATH": video_path,
            "TARGET_MARKER_SECS": target_marker_secs,
            "MARKER_COLOR": marker_color,
            "MARKER_NAME": marker_name,
            "MARKER_NOTE": marker_note,
            "MARKER_DURATION": 10,
        }
        return json.dumps(payload).encode("utf-8")
    else:
        return b"shutdown"

# TODO - add logic to shutdown resolve controller
def send_payload_to_resolve(
    video_path,
    target_marker_secs,
    marker_color,
    marker_name,
    marker_note,
    exit_command=False,
):
    payload = prepare_payload(
        video_path,
        target_marker_secs,
        marker_color,
        marker_name,
        marker_note,
        exit_command,
    )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", 65432))
            s.sendall(payload)
            response_data = s.recv(4096)
            response = json.loads(response_data.decode("utf-8"))
            return response.get("status") == 200
    except Exception:
        return False


# Создаем UI
with ui.header().classes("justify-between text-white bg-slate-800"):
    ui.label("Анализ видео").classes("text-2xl font-bold")
    ui.dark_mode().bind_value(ui.query("body"), "dark")

with ui.left_drawer().classes("bg-slate-900 p-4 w-64"):
    ui.label("⚙️ Настройки").classes("text-xl font-bold mb-4")

    # Выбор папки с видео
    video_settings = (
        ui.row()
        .classes("items-center w-full")
        .bind_visibility_from(state, "show_video_settings")
    )
    with video_settings:
        video_dir_input = (
            ui.input("Путь к папке")
            .bind_value_to(state, "video_dir")
            .classes("flex-grow")
        )
        ui.button(icon="folder", on_click=select_folder).props(
            "color=accent text-color=white rounded"
        )
        mount_button = (
            ui.button("Монтировать папку", on_click=mount_and_list, icon="folder_open")
            .classes("w-full")
            .props("color=accent text-color=white rounded")
        )
        mount_button.bind_visibility_from(state, "show_mount_button")
        extract_button = (
            ui.button(
                "Извлечь кадры",
                on_click=extract_frames_and_embeddings,
                icon="movie_filter",
            )
            .classes("w-full mt-4")
            .props("color=green text-color=white rounded")
        )
        extract_button.bind_visibility_from(state, "show_extract_button")

    # Настройки OpenRouter
    ui.label("OpenRouter API").classes("text-lg font-bold mt-4")
    ui.input("API ключ").bind_value_to(state, "openrouter_user_api_key").classes(
        "w-full"
    )
    ui.input("Модель").bind_value_to(state, "openrouter_user_model").classes("w-full")

with ui.tabs().classes("w-full mt-4") as tabs:
    query_tab = ui.tab("Поиск по видео", icon="search")
    report_tab = ui.tab("Отчет", icon="description")

with ui.tab_panels(tabs, value=query_tab).classes("w-full"):
    with ui.tab_panel(query_tab):
        ui.label("Поиск по видео").classes("text-lg mb-4")
        with ui.row().classes("w-full items-center"):
            ui.input("Поисковый запрос").bind_value_to(state, "query_text").classes(
                "flex-grow"
            ).props("rounded outlined dense")
            num_results_input = (
                ui.number(min=1, max=18, value=6)
                .props("dense outlined rounded hide-bottom-space")
                .classes("w-20")
            )
            num_results_input.bind_value_to(state, "k")
            ui.button(
                "Искать", on_click=lambda: query_similar_images(state.k), icon="search"
            ).classes("ml-2").props("color=accent text-color=white rounded")
            ui.button(
                "RESOLVE",
                on_click=lambda: send_payload_to_resolve(
                    video_path="/Users/aleko/Documents/elbrus_bootcamp/ds-phase-3/_FINAL_PROJECT/wedding_stream.mp4",
                    target_marker_secs=120,
                    marker_color="Lemon",
                    marker_name="www",
                    marker_note="yyy",
                    exit_command=False,
                ),
                icon="search",
            ).classes("ml-2").props("color=accent text-color=white rounded")

        results_container = ui.column().classes("w-full mt-4 gap-4")

    with ui.tab_panel(report_tab):
        ui.label("Генерация отчета").classes("text-lg mb-4")
        ui.button(
            "Создать PDF", on_click=generate_pdf_report, icon="picture_as_pdf"
        ).classes("w-full")

        open_pdf_button = ui.button(
            "Открыть отчет", on_click=open_pdf, icon="open_in_new"
        ).classes("w-full mt-4")
        open_pdf_button.visible = False

ui.run(
    title="Анализ видео", reload=False, native=True, port=8001, window_size=(1200, 800)
)
