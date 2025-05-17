import os
import socket
import json
import sys
import time
import cv2

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
            "Project's media folder doesn't contain any clips, we'll try importing..."
        )

    target_filename = os.path.basename(VIDEO_PATH)

    # Check if clip already exists
    clip = None
    for c in clips:
        if c.GetName() == target_filename:
            clip = c
            print(f"Clip '{target_filename}' already exists in Media Pool.")
            break

    # If not found, import it
    if not clip:
        print(f"📁 Importing '{target_filename}' into Media Pool...")
        imported_items = mediaPool.ImportMedia([VIDEO_PATH])
        if not imported_items or not imported_items[0]:
            print("❌ Failed to import video file.")
            return {"status": 500, "error": "Failed to import video file."}
        clip = imported_items[0]
        print("✅ Successfully imported.")

    # Confirm we have a valid clip
    frames_property = clip.GetClipProperty("Frames")
    if not frames_property:
        print("⚠️ Unable to get clip 'Frames' property.")
        return {"status": 500, "error": "Clip property 'Frames' missing."}
    num_frames = int(frames_property)
    print(f"Clip '{clip.GetName()}' has {num_frames} frames.")

    # Get clip's FPS to calculate the correct frame
    clip_fps = clip.GetClipProperty().get("FPS")
    target_marker_frame = round(float(clip_fps) * TARGET_MARKER_SECS)

    # Add Markers
    if num_frames >= target_marker_frame:
        isSuccess = clip.AddMarker(
            target_marker_frame, MARKER_COLOR, MARKER_NAME, MARKER_NOTE, MARKER_DURATION
        )
        if isSuccess:
            print(
                f"✅ Added marker at {TARGET_MARKER_SECS}s, FrameId:{target_marker_frame}"
            )
            return {"status": 200}
        else:
            print("❌ Failed to add marker. Maybe it already exists?")
            return {
                "status": 500,
                "error": "Failed to add marker. It may already exist.",
            }
    else:
        print("⚠️ Marker frame exceeds clip length.")
        return {"status": 500, "error": "Marker frame exceeds clip length."}


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
                conn.sendall(
                    json.dumps(
                        {"status": 200, "message": "Controller shutting down."}
                    ).encode("utf-8")
                )
                conn.close()
                break

            try:
                payload = json.loads(message)
                try:
                    response = handle_marker_request(payload)
                    if not response:
                        response = {"status": 500, "error": "No response from handler."}
                except Exception as e:
                    print(f"⚠️ Handler error: {e}")
                    response = {"status": 500, "error": str(e)}
            except Exception as e:
                print(f"⚠️ Error parsing payload: {e}")
                response = {"status": 400, "error": f"Invalid JSON payload: {e}"}

            # Send response
            try:
                conn.sendall(json.dumps(response).encode("utf-8"))
            except Exception as e:
                print(f"⚠️ Error sending response: {e}")

            conn.close()

    except KeyboardInterrupt:
        print("🛑 Script interrupted manually.")
        break

server_socket.close()
print("👋 Socket closed. Goodbye!")
sys.exit()
