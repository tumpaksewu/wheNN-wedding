import os
import socket
import json
import sys
import time

# === SOCKET CONFIGURATION ===
HOST = "localhost"
PORT = 65432
BUFFER_SIZE = 4096


# Handle incoming marker request
def handle_marker_request(payload):
    global project

    VIDEO_PATH = payload.get("VIDEO_PATH")
    TARGET_MARKER_SECS = int(payload.get("TARGET_MARKER_SECS", 0))
    MARKER_COLOR = payload.get("MARKER_COLOR", "Red")
    MARKER_NAME = payload.get("MARKER_NAME", "First Item")
    MARKER_NOTE = payload.get("MARKER_NOTE", "Second Item")
    MARKER_DURATION = int(payload.get("MARKER_DURATION", 10))

    print(f"\n📩 Received request for marker at {TARGET_MARKER_SECS}s.")

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ File not found: {VIDEO_PATH}")
        return

    TARGET_MARKER_FRAME = int(
        project.GetSetting("timelineFrameRate") * TARGET_MARKER_SECS
    )

    # Open Media page
    resolve.OpenPage("media")

    # Get supporting objects
    projectManager = resolve.GetProjectManager()
    mediaPool = project.GetMediaPool()
    rootBin = mediaPool.GetRootFolder()

    # Go to root bin
    mediaPool.SetCurrentFolder(rootBin)

    # Gets clips
    clips = rootBin.GetClipList()
    if not clips or not clips[0]:
        print(
            "Error: MediaPool root bin doesn't contain any clips, add one clip (recommended clip duration >= 80 frames) and try again!"
        )

    target_filename = os.path.basename(VIDEO_PATH)

    # Check if clip already exists
    clip = None
    for c in clips:
        if c.GetName() == target_filename:
            clip = c
            print(f"📎 Clip '{target_filename}' already exists in Media Pool.")
            break

    # If not found, import it
    if not clip:
        print(f"📁 Importing '{target_filename}' into Media Pool...")
        imported_items = mediaPool.ImportMedia([VIDEO_PATH])
        if not imported_items or not imported_items[0]:
            print("❌ Failed to import video file.")
            return
        clip = imported_items[0]
        print("✅ Successfully imported.")

    # Confirm we have a valid clip
    frames_property = clip.GetClipProperty("Frames")
    if not frames_property:
        print("⚠️ Unable to get clip 'Frames' property.")
        return
    num_frames = int(frames_property)
    print(f"🎞️ Clip '{clip.GetName()}' has {num_frames} frames.")

    # Add Markers
    if num_frames >= TARGET_MARKER_FRAME:
        isSuccess = clip.AddMarker(
            TARGET_MARKER_FRAME, MARKER_COLOR, MARKER_NAME, MARKER_NOTE, MARKER_DURATION
        )
        if isSuccess:
            print(f"✅ Added marker at FrameId:{TARGET_MARKER_FRAME}")
        else:
            print("❌ Failed to add marker")
    else:
        print("⚠️ Marker frame exceeds clip length.")


# Create non-blocking socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
server_socket.setblocking(0)  # Non-blocking
print(f"📡 Socket server is ready on {HOST}:{PORT}...")

# === MAIN SCRIPT ===

project = resolve.GetProjectManager().GetCurrentProject()

if not project:
    print("No project is loaded")
    sys.exit()

print("🎬 Script initialized successfully. Waiting for socket input...")

# Main loop to poll for socket connections
while True:
    try:
        conn = None
        try:
            conn, addr = server_socket.accept()
            print(f"🔌 Connected by {addr}")
        except BlockingIOError:
            # No connection yet
            time.sleep(0.1)
            continue

        if conn:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                conn.close()
                continue

            message = data.decode("utf-8").strip()

            if message == "shutdown":
                print("🛑 Shutdown command received. Exiting script.")
                conn.close()
                break

            try:
                payload = json.loads(message)
                handle_marker_request(payload)
            except Exception as e:
                print(f"⚠️ Error parsing payload: {e}")

            conn.close()

    except KeyboardInterrupt:
        print("🛑 Script interrupted manually.")
        break

server_socket.close()
print("👋 Socket closed. Goodbye!")
sys.exit()
