import socket
import json
import time


def resolve_import_and_seek(data):
    global resolve
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if not project:
        print("❌ No active project.")
        return

    media_pool = project.GetMediaPool()
    if not media_pool:
        print("❌ Could not access Media Pool.")
        return

    def find_clip(folder, filename):
        for clip in folder.GetClipList():
            if clip.GetName() == filename:
                return clip
        for subfolder in folder.GetSubFolderList():
            found = find_clip(subfolder, filename)
            if found:
                return found
        return None

    filename = data["filename"]
    timestamp = data["timestamp"]  # seconds

    # Step 1: Check if file exists in media pool
    root_folder = media_pool.GetRootFolder()
    clip = find_clip(root_folder, filename)

    if not clip:
        print(f"Importing {filename}...")
        media_storage = resolve.GetMediaStorage()
        clips = media_storage.AddItemListToMediaPool(filename)
        if not clips:
            print("❌ Failed to import file.")
            return
        clip = clips[0]

    # Step 2: Switch to Cut Page
    print("Switching to Cut Page...")
    resolve.OpenPage("cut")

    # Step 3: Load the clip into the Cut Page viewer
    print(f"Loading clip '{clip.GetName()}' into Cut Page player...")
    media_player = resolve.GetMediaPlayer()
    success = media_player.LoadClipIntoPlayer(clip)
    if not success:
        print("❌ Failed to load clip into player.")
        return

    # Step 4: Seek to timestamp
    frame_rate = project.GetSettings()["timelineFrameRate"]
    frame_number = int(timestamp * frame_rate)
    timecode_str = f"@{frame_number}"
    print(f"Seeking to frame: {frame_number} ({timecode_str})")

    success = project.SetCurrentTimecode(timecode_str)
    if not success:
        print("❌ Failed to seek to timecode.")
    else:
        print(f"✅ Seeked to {timestamp}s ({frame_number} frames)")


# Socket server setup
def start_server():
    host, port = "localhost", 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("Listening for connections...")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                try:
                    data = json.loads(conn.recv(1024).decode())
                    resolve_import_and_seek(data)
                except Exception as e:
                    print("Error processing request:", e)


if __name__ == "__main__":
    start_server()