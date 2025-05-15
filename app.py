from nicegui import ui, app
import os

# Хранилище загруженных видео
VIDEO_DATA = []

# Папка для временного хранения ссылок на файлы
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Статические файлы
app.add_static_files('/upload', UPLOAD_DIR)


def handle_upload(e):
    """Обработчик загрузки видео"""
    file_path = os.path.join(UPLOAD_DIR, e.name)
    e.content.seek(0)
    with open(file_path, "wb") as f:
        f.write(e.content.read())

    VIDEO_DATA.append({
        'name': e.name,
    })

    update_dropdown_options()
    ui.notify(f"Видео '{e.name}' загружено")


def update_dropdown_options():
    """Обновляем список только с именами файлов"""
    video_dropdown.options = [item['name'] for item in VIDEO_DATA]
    video_dropdown.update()


def select_video(e):
    selected_name = e.value
    for item in VIDEO_DATA:
        if item['name'] == selected_name:
            video_player.set_source(f"/upload/{selected_name}")
            ui.notify(f"Воспроизводится: {selected_name}")
            return


def seek_to(seconds):
    """Перемотка видео"""
    ui.run_javascript(f'''
        const videoElement = document.querySelector("video");
        if (videoElement) {{
            videoElement.currentTime = {seconds};
        }}
    ''')


# === Интерфейс ===

video_player = ui.video(src="").classes("w-full")
video_player.controls = True
video_player.set_source("http://localhost:8000/videos/wedding-raw-footage.mp4")

ui.label("Выберите видеофайлы").classes("text-2xl")

ui.upload(
    multiple=True,
    on_upload=handle_upload,
    label="Загрузить видео"
).classes("w-full")

video_dropdown = ui.select(
    label="Видео",
    options=[],
    on_change=select_video
).classes("mt-4 w-full")

with ui.row().classes("w-full justify-between mt-4"):
    ui.button("0:01", on_click=lambda: seek_to(1)).props("flat")
    ui.button("0:05", on_click=lambda: seek_to(5)).props("flat")
    ui.button("0:10", on_click=lambda: seek_to(10)).props("flat")

ui.run()





# from nicegui import ui
# import os

# # Создаем папку для загрузок
# os.makedirs("uploads", exist_ok=True)

# def handle_upload(e):
#     """Обработчик загрузки видео"""
#     file_path = f"uploads/{e.name}"
#     # Записываем содержимое файла правильно
#     e.content.seek(0)  # убедимся, что курсор в начале
#     with open(file_path, "wb") as f:
#         f.write(e.content.read())
    
#     # Обновляем источник видео через set_source
#     video.set_source(file_path)
#     ui.notify(f"Видео {e.name} загружено!")

# def seek_video():
#     timestamp = input_timestamp.value
#     try:
#         seconds = float(timestamp)
#         if seconds >= 0:
#             # Устанавливаем время воспроизведения
#             ui.run_javascript(f'''
#                 const videoElement = document.querySelector("video");
#                 if (videoElement) {{
#                     videoElement.currentTime = {seconds};
#                 }}
#             ''')
#         else:
#             ui.notify("Введите положительное число!", color="negative")
#     except ValueError:
#         ui.notify("Введите корректное число!", color="negative")

# # Создаем видео-плеер с пустым источником
# video = ui.video(src="").classes("w-full")
# video.autoplay = False
# video.controls = True

# # Интерфейс
# ui.label("Загрузите видео").classes("text-2xl")
# ui.upload(on_upload=handle_upload).classes("w-full")

# with ui.row().classes("mt-4"):
#     input_timestamp = ui.input(label="Перейти к секунде", placeholder="Например: 30")
#     ui.button("Перейти", on_click=seek_video).props("flat")

# ui.run()











