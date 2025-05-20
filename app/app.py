from nicegui import run, ui
import httpx
import json
import csv
import socket
import os
import subprocess
import sys
import webbrowser
from datetime import timedelta, datetime
from typing import Optional
import tkinter as tk
from tkinter import filedialog

# macOS packaging support
from multiprocessing import freeze_support  # noqa

freeze_support()  # noqa

from picker import select_folder


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


API_URL = f"http://{get_local_ip()}:8000"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(SCRIPT_DIR, "settings.json")
SEARCH_HISTORY_PATH = os.path.join(SCRIPT_DIR, "search_history.csv")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}


class AppState:
    def __init__(self):
        self.video_dir: Optional[str] = ""
        self.selected_video: Optional[str] = None
        self.openrouter_user_api_key: str = ""
        self.openrouter_user_model: str = ""
        self.query_text: str = ""
        self.query_results = []
        self.progress_message: str = ""
        self.show_mount_button = False
        self.show_extract_button = False
        self.show_video_settings = True
        self.k = 6
        self.resolve_controller_enabled = False
        self.resolve_switch_active = False
        self.marker_color = "Red"


state = AppState()

ui.colors(
    primary="purple",
    secondary="#2A004E",
    dark="#2b2b2b",
    dark_page="#181818",
)

ui.add_head_html("""
<style>
.q-dialog__backdrop {
    background-color: rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(8px);
}
</style>
""")
ui.add_head_html(
    '<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />'
)


def save_settings_to_file():
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(
                {
                    "video_dir": state.video_dir,
                    "openrouter_user_api_key": state.openrouter_user_api_key,
                    "openrouter_user_model": state.openrouter_user_model,
                },
                f,
            )
            ui.notify("Настройки OpenRouter сохранены!", type="positive")
    except Exception as e:
        ui.notify(f"Could not save settings: {e}", type="negative")


def load_settings_from_file():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r") as f:
                settings = json.load(f)
                state.video_dir = settings.get("video_dir", "")
                state.openrouter_user_api_key = settings.get(
                    "openrouter_user_api_key", ""
                )
                state.openrouter_user_model = settings.get("openrouter_user_model", "")
        except Exception as e:
            ui.notify(f"Could not load settings: {e}", type="warning")


async def handle_folder_click():
    result, e = await run.cpu_bound(select_folder)
    if result:
        state.show_mount_button = True
        state.video_dir = result
        video_dir_input.value = result
        ui.notify(f"Выбрана папка: {result}", type="positive")
    else:
        ui.notify(f"Ошибка при выборе папки: {str(e)}", type="negative")


# def select_folder():
#     e = None
#     root = tk.Tk()
#     root.withdraw()
#     root.wm_attributes("-topmost", 1)

#     try:
#         folder = filedialog.askdirectory(title="Выберите папку с видео")
#         if folder:
#             return folder, e
#     except Exception as e:
#         return None, e


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

        save_settings_to_file()
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
                            progress_area.push("✅ Завершено.")
                            state.show_video_settings = False
                            ui.notify("Обработка завершена", type="positive")
                            break

                        # Add line to UI
                        progress_area.push(message)

        progress_dialog.close()

    except Exception as e:
        progress_area.push(f"❌ Ошибка: {str(e)}")
        ui.notify(f"Ошибка: {str(e)}", type="negative")
        progress_dialog.close()


async def query_similar_images(k: int, history_query=None):
    if history_query:
        state.query_text = history_query
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

        log_search_query(state.query_text)
        update_history_drawer()

        ui.notify(f"Найдено {len(state.query_results)} результатов", type="positive")
    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}", type="negative")


def log_search_query(query: str):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    file_exists = os.path.isfile(SEARCH_HISTORY_PATH)

    with open(SEARCH_HISTORY_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "query"])
        writer.writerow([timestamp, query])


async def handle_generate_scene_captions():
    spinner.visible = True
    try:
        async with httpx.AsyncClient(timeout=240) as client:
            response = await client.get(f"{API_URL}/generate_scene_captions")
            response.raise_for_status()
            ui.notify("Сцены успешно сгенерированы!", type="positive")
    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}", type="negative")
    finally:
        spinner.visible = False


async def generate_pdf_report():
    spinner.visible = True
    try:
        async with httpx.AsyncClient(timeout=240) as client:
            response = await client.post(
                f"{API_URL}/generate_pdf_report",
                json={
                    "openrouter_api_key": state.openrouter_user_api_key,
                    "openrouter_model": state.openrouter_user_model,
                },
            )

        response.raise_for_status()

        pdf_path = "scenario_report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        ui.notify("PDF отчет доступен для просмотра!", type="positive")
        open_pdf_button.visible = True
    except Exception as e:
        ui.notify(f"Ошибка: {str(e)}. Попробуйте еще раз.", type="negative")
    finally:
        spinner.visible = False


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
        # FIXME - find some use for marker_note
        success = send_payload_to_resolve(
            video_path=f"{state.video_dir}/{video_name}",
            target_marker_secs=timestamp,
            marker_color=state.marker_color,
            marker_name=state.query_text,
            marker_note="testttt",
        )

        if success:
            ui.notify("✅ Маркер добавлен!", color="positive")
        else:
            ui.notify(
                "❌ Не удалось добавить маркер. Возможно, он уже существует?",
                color="negative",
            )

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

                with (
                    ui.card()
                    .tight()
                    .classes(
                        "rounded-lg w-64 relative transition-all duration-200 hover:shadow-xl hover:scale-105"
                    )
                    .props("flat bordered")
                ):
                    ui.image(f"{API_URL}/frames/{image_name}").classes(
                        "w-full rounded-t-lg cursor-pointer"
                    ).on(
                        "click",
                        lambda e, r=result: show_video_preview(
                            r["video_name"], r["timestamp"]
                        ),
                    )

                    ui.button(
                        "",
                        icon="push_pin",
                        on_click=create_on_click_mrk_button(video_name, timestamp),
                    ).props("round dense flat").classes(
                        "absolute top-2 right-2 z-10 backdrop-blur-sm text-white hover:bg-black/50"
                    ).bind_visibility_from(state, "resolve_controller_enabled")

                    with ui.card_section():
                        ui.label(f"🎞 Видео: {video_name}")
                        ui.label(f"🕑 Время: {str(timedelta(seconds=int(timestamp)))}")
                        ui.label(f"🔍 Сходство: {score:.2f}")


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


def is_resolve_controller_active(host="127.0.0.1", port=65432, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            state.resolve_controller_enabled = True
            ui.notify("Установлено подключение к Resolve Controller", type="positive")
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        state.resolve_controller_enabled = False
        ui.notify("Не удалось подключиться к Resolve Controller", type="negative")
        return False


async def handle_resolve_switch(e):
    switch_position = e.args[0]
    if switch_position:
        # User is trying to turn ON the controller
        if is_resolve_controller_active():
            ui.notify("Контроллер включён", type="positive")
            state.resolve_controller_enabled = True
        else:
            # If controller not available, revert switch back to OFF
            ui.notify("Не удалось подключиться к Resolve Controller", type="negative")
            state.resolve_switch_active = False
    else:
        # User is turning OFF
        disable_controller()


def disable_controller():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", 65432))
        s.sendall(b"shutdown")
    state.resolve_switch_active = False
    state.resolve_controller_enabled = False
    ui.notify("Контроллер отключён", color="orange")


def send_payload_to_resolve(
    video_path,
    target_marker_secs,
    marker_color,
    marker_name,
    marker_note,
):
    payload = {
        "VIDEO_PATH": video_path,
        "TARGET_MARKER_SECS": target_marker_secs,
        "MARKER_COLOR": marker_color,
        "MARKER_NAME": marker_name,
        "MARKER_NOTE": marker_note,
        "MARKER_DURATION": 10,
    }
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", 65432))
            s.sendall(json.dumps(payload).encode("utf-8"))
            response_data = s.recv(4096)
            response = json.loads(response_data.decode("utf-8"))
            return response.get("status") == 200
    except Exception:
        return False


def history_helper():
    settings_drawer.hide()
    history_drawer.toggle()


# Function to load history from CSV
def load_search_history():
    history = []
    try:
        with open(SEARCH_HISTORY_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
    except FileNotFoundError:
        pass
    return history


# Function to update drawer UI with reversed history
def update_history_drawer():
    timeline.clear()
    history = load_search_history()
    for entry in reversed(history):
        timestamp = entry["timestamp"]
        query = entry["query"]
        with timeline:
            with ui.timeline_entry().props("side='right'"):
                history_entry = (
                    ui.chip(query, icon="search")
                    .props("clickable text-color=white")
                    .classes("hover:shadow-xl hover:scale-110")
                )

                def make_click_handler(q):
                    async def handler(_):
                        history_drawer.hide()
                        await query_similar_images(state.k, q)

                    return handler

                history_entry.on("click", make_click_handler(query))
                ui.label(timestamp).classes("text-sm text-gray-500")


def update_k(delta: int):
    state.k = max(1, min(18, state.k + delta))  # clamp between 1 and 18


# MAIN UI —————————————————————————————
load_settings_from_file()

with ui.header().classes("bg-secondary flex justify-between items-center px-4"):
    # LEFT SECTION
    with ui.row().classes("items-center gap-4 flex-nowrap"):
        ui.button(on_click=lambda: settings_drawer.toggle(), icon="menu").props(
            "outline rounded color=slate-20"
        )
        with ui.column().classes("leading-none justify-center -space-y-1"):
            ui.label("wheNN [wedding]").classes("text-sm font-bold leading-none")
            ui.label("> Build 1.4.1").classes("text-xs opacity-70 leading-none")

        with ui.tabs().classes("ml-4") as tabs:
            query_tab = ui.tab("Поиск по видео", icon="search")
            report_tab = ui.tab("Отчет", icon="description")

    # RIGHT SECTION
    with ui.row().classes("items-center gap-4"):
        spinner = (
            ui.spinner(size="lg", color="white")
            .props('thickness="4"')
            .classes("q-ma-md")
        )
        spinner.visible = False
        spinner
        ui.switch("").props(
            "dark=true checked-icon=dark_mode unchecked-icon=light_mode"
        ).bind_value(ui.dark_mode()).classes("text-white")
        ui.button(
            icon="eva-github",
            on_click=lambda: webbrowser.open(
                "https://github.com/tumpaksewu/wheNN-wedding"
            ),
        ).props("outline round text-color=white")
        ui.button(on_click=lambda: history_helper(), icon="history").props(
            "outline round color=slate-20"
        )


with ui.left_drawer().classes("p-4 w-64 shadow-lg") as settings_drawer:
    ui.label("⚙️ Настройки").classes("text-xl font-bold mb-4")
    ui.separator()

    # Выбор папки с видео
    ui.label("Рабочая папка").classes("text-lg font-bold mt-4")
    video_settings = (
        ui.row()
        .classes("items-center w-full")
        .bind_visibility_from(state, "show_video_settings")
    )
    with video_settings:
        video_dir_input = (
            ui.input("Путь к папке", value=state.video_dir)
            .bind_value_to(state, "video_dir")
            .classes("flex-grow")
        )
        ui.button(icon="folder", on_click=handle_folder_click).props(
            "text-color=white rounded"
        )
        mount_button = (
            ui.button(
                "Монтировать папку", on_click=mount_and_list, icon="drive_folder_upload"
            )
            .classes("w-full")
            .props("text-color=white rounded")
        )
        # mount_button.bind_visibility_from(state, "show_mount_button")
        mount_button.bind_visibility_from(state, "video_dir")
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
    ui.input("API ключ", value=state.openrouter_user_api_key).bind_value_to(
        state, "openrouter_user_api_key"
    ).classes("w-full")
    ui.input("Модель", value=state.openrouter_user_model).bind_value_to(
        state, "openrouter_user_model"
    ).classes("w-full")
    ui.button("Сохранить", on_click=save_settings_to_file).classes("w-full").props(
        "text-color=white rounded"
    )

    # Настройки Resolve Controller
    ui.label("Resolve Controller").classes("text-lg font-bold mt-4")
    with ui.row():
        resolve_switch = ui.switch("Resolve Controller").bind_value(
            state, "resolve_switch_active"
        )
        resolve_switch.on("update:model-value", handle_resolve_switch)

drawer_toggle = ui.element()
with (
    ui.right_drawer(value=False).bind_value(drawer_toggle).classes("p-4 w-64 shadow-lg")
) as history_drawer:
    ui.label("История").classes("text-xl font-bold mb-4")
    ui.separator()
    with ui.timeline(side="right") as timeline:
        pass

with ui.tab_panels(tabs, value=query_tab).classes("w-full rounded-lg shadow-md"):
    with ui.tab_panel(query_tab):
        with ui.row().classes("w-full items-center"):
            ui.input("Поисковый запрос").bind_value_to(state, "query_text").classes(
                "flex-grow h-14"
            ).props("clearable rounded outlined color=grey-7")
            # Interactive num_results selector
            with ui.row().classes("items-center"):
                ui.button(icon="chevron_left", on_click=lambda: update_k(-1)).props(
                    "flat round color=grey-7"
                ).classes("w-14 h-14")

                value_display = ui.label(f"{state.k}").classes(
                    "text-xl w-10 text-center"
                )

                def refresh_label():
                    value_display.text = str(state.k)

                ui.button(icon="chevron_right", on_click=lambda: update_k(1)).props(
                    "flat round color=grey-7"
                ).classes("w-14 h-14")

                # Auto-update display when state.k changes
                ui.timer(0.1, refresh_label)

            ui.button(
                "Искать", on_click=lambda: query_similar_images(state.k), icon="search"
            ).classes("ml-2 h-14 px-4 text-lg").props("text-color=white rounded")

        results_container = ui.column().classes("w-full mt-4 gap-4")

        # Resolve marker color picker
        with ui.page_sticky(x_offset=18, y_offset=18):
            marker_color_picker = ui.element("q-fab").props(
                "color=primary icon=palette direction=left"
            )
            with marker_color_picker:
                ui.element("q-fab-action").props("icon=circle color=red-5").on(
                    "click", lambda: setattr(state, "marker_color", "Red")
                )
                ui.element("q-fab-action").props("icon=circle color=green-5").on(
                    "click", lambda: setattr(state, "marker_color", "Green")
                )
                ui.element("q-fab-action").props("icon=circle color=blue-5").on(
                    "click", lambda: setattr(state, "marker_color", "Blue")
                )
                ui.element("q-fab-action").props("icon=circle color=cyan-5").on(
                    "click", lambda: setattr(state, "marker_color", "Cyan")
                )
                ui.element("q-fab-action").props("icon=circle color=pink-5").on(
                    "click", lambda: setattr(state, "marker_color", "Pink")
                )
                ui.element("q-fab-action").props("icon=circle color=lime-5").on(
                    "click", lambda: setattr(state, "marker_color", "Lemon")
                )

    marker_color_picker.bind_visibility_from(state, "resolve_controller_enabled")

    with ui.tab_panel(report_tab):
        ui.button(
            "Сгенерировать сцены",
            on_click=handle_generate_scene_captions,
            icon="subtitles",
        ).classes("w-full").props("text-color=white rounded")

        ui.button(
            "Создать PDF", on_click=generate_pdf_report, icon="picture_as_pdf"
        ).classes("w-full").props("text-color=white rounded")

        open_pdf_button = (
            ui.button("Открыть отчет", on_click=open_pdf, icon="open_in_new")
            .classes("w-full mt-4")
            .props("color=green text-color=white rounded")
        )
        open_pdf_button.visible = False

    update_history_drawer()

ui.run(
    title="wheNN[wedding]",
    reload=False,
    native=True,
    host="0.0.0.0",
    port=8001,
    window_size=(1200, 800),
)
