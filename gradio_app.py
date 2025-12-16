# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import sys
# import asyncio

# # --- Windows asyncio fix to reduce Proactor / WinError 10054 noise ---
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# import cv2
# import gradio as gr
# from ultralytics import YOLO
# from pathlib import Path
# import tempfile
# import time
# import torch
# import subprocess
# import shutil

# # -----------------------------
# # Basic CUDA info (sanity check)
# # -----------------------------
# print(torch.__version__)
# print("is_available:", torch.cuda.is_available())
# print("device_count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("current device:", torch.cuda.current_device())
#     print("device name:", torch.cuda.get_device_name(0))

# DEVICE = "0" if torch.cuda.is_available() else "cpu"  # used for CLI

# # -----------------------------
# # Load model for IMAGE inference
# # (we'll keep this in Python, it’s already fast)
# # -----------------------------
# MODEL_PATH = "runs/detect/dog_breed_yolo11n5/weights/best.pt"
# model = YOLO(MODEL_PATH)
# print("Model device reported by Ultralytics (initial):", model.device)


# # -----------------------------
# # Image prediction (Python)
# # -----------------------------
# def predict_image(image):
#     """
#     Run YOLO on a single image (numpy array) and return image with bounding boxes.
#     Gradio passes images as RGB numpy arrays by default.
#     """
#     if image is None:
#         return None

#     # Force device here; in practice this is already fast
#     results = model.predict(image, device=DEVICE, verbose=False)

#     if results:
#         print("Image speed (ms):", results[0].speed)

#     plotted = results[0].plot()  # RGB
#     return plotted


# # -----------------------------
# # Video prediction (CLI -> GPU)
# # -----------------------------
# def predict_video(video_file):
#     """
#     Use the Ultralytics CLI in a subprocess for video, forcing device=0.
#     This sidesteps whatever is causing the Python predictor to stick to CPU.
#     """
#     if video_file is None:
#         return None

#     # Gradio may pass a dict or a plain path string
#     if isinstance(video_file, dict):
#         input_path = video_file.get("name") or video_file.get("path")
#     else:
#         input_path = video_file

#     if input_path is None:
#         print("No video path from Gradio input.")
#         return None

#     input_path = str(input_path)
#     if not os.path.isfile(input_path):
#         print("Could not find video file:", input_path)
#         return None

#     # Ensure a predictable project folder
#     project_dir = Path("gradio_runs")
#     project_dir.mkdir(parents=True, exist_ok=True)

#     # Unique run name
#     run_name = f"gradio_vid_{int(time.time())}"

#     # Build CLI command
#     # This is equivalent to running:
#     # yolo predict model=... source=... device=0 save=True project=gradio_runs name=...
#     cmd = [
#         "yolo",
#         "predict",
#         f"model={MODEL_PATH}",
#         f"source={input_path}",
#         f"device={DEVICE}",       # "0" for GPU, "cpu" otherwise
#         "save=True",
#         f"project={project_dir}",
#         f"name={run_name}",
#         "exist_ok=True",
#     ]

#     print("[INFO] Running CLI:", " ".join(cmd))
#     t0 = time.time()
#     try:
#         # Run the command and wait for it to finish
#         completed = subprocess.run(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             check=False,
#         )
#     except FileNotFoundError:
#         print("[ERROR] 'yolo' command not found. Make sure Ultralytics CLI is installed.")
#         return None

#     t1 = time.time()
#     print("[INFO] CLI finished in {:.2f}s".format(t1 - t0))
#     print("[CLI OUTPUT]")
#     print(completed.stdout)

#     # Find the output video
#     save_dir = project_dir / run_name
#     if not save_dir.exists():
#         print("[ERROR] Expected output dir does not exist:", save_dir)
#         return None

#     # Look for an mp4 in the save_dir
#     mp4s = list(save_dir.glob("*.mp4"))
#     if not mp4s:
#         print("[ERROR] No .mp4 output found in", save_dir)
#         return None

#     out_path = mp4s[0]
#     print("[INFO] Output video:", out_path)

#     # (Optional) Move it to a temp file so it’s not in runs forever
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     tmp_path = Path(tmp.name)
#     tmp.close()
#     shutil.copy2(out_path, tmp_path)

#     return str(tmp_path)


# # -----------------------------
# # Gradio UI
# # -----------------------------
# with gr.Blocks() as demo:
#     gr.Markdown("# 🐶 Dog Breed YOLO Detector (GPU, CLI Video Pipeline)")
#     gr.Markdown(
#         "Upload an **image** or an **MP4 video**.\n\n"
#         "- **Image**: uses `model.predict(image, device=DEVICE)` in Python.\n"
#         "- **Video**: calls the Ultralytics **CLI** (`yolo predict ... device=0`) in a subprocess, "
#         "which is the same fast path you’d use from the terminal.\n"
#         "Check the console logs for the total runtime reported by the CLI."
#     )

#     # ---- Image tab ----
#     with gr.Tab("Image"):
#         with gr.Row():
#             with gr.Column():
#                 image_input = gr.Image(type="numpy", label="Upload an image")
#                 image_button = gr.Button("Run detection")
#             with gr.Column():
#                 image_output = gr.Image(type="numpy", label="Detected image")

#         image_button.click(fn=predict_image, inputs=image_input, outputs=image_output)

#     # ---- Video tab ----
#     with gr.Tab("Video (MP4)"):
#         with gr.Row():
#             with gr.Column():
#                 video_input = gr.Video(label="Upload a video (MP4)")
#                 video_button = gr.Button("Run video detection")
#             with gr.Column():
#                 video_output = gr.Video(label="Processed video")

#         video_button.click(fn=predict_video, inputs=video_input, outputs=video_output)


# if __name__ == "__main__":
#     demo.launch()

#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import asyncio

# --- Windows asyncio fix to reduce Proactor / WinError 10054 noise ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import tempfile
import time
import torch
import subprocess
import shutil




# -----------------------------
# Basic CUDA info (sanity check)
# -----------------------------
print(torch.__version__)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))

DEVICE = "0" if torch.cuda.is_available() else "cpu"  # used for CLI

IMG_SIZE = 640

# -----------------------------
# Load model (still useful for sanity, but not used for image inference now)
# -----------------------------
MODEL_PATH = "runs/detect/dog_breed_yolo11n5/weights/best.pt"
model = YOLO(MODEL_PATH)
print("Model device reported by Ultralytics (initial):", model.device)


# -----------------------------
# Image prediction (CLI -> GPU, same pattern as video)
# -----------------------------
def predict_image(image_file):
    """
    Use the Ultralytics CLI in a subprocess for images, forcing device=DEVICE.
    Returns the path to the processed image for Gradio to display.
    """
    if image_file is None:
        return None

    # Gradio may pass a dict or a plain path string
    if isinstance(image_file, dict):
        input_path = image_file.get("name") or image_file.get("path")
    else:
        input_path = image_file

    if input_path is None:
        print("[ERROR] No image path from Gradio input.")
        return None

    input_path = str(input_path)
    if not os.path.isfile(input_path):
        print("[ERROR] Could not find image file:", input_path)
        return None

    # Ensure a predictable project folder
    project_dir = Path("gradio_runs")
    project_dir.mkdir(parents=True, exist_ok=True)

    # Unique run name
    run_name = f"gradio_img_{int(time.time())}"

    # Build CLI command
    # cmd = [
    #     "yolo",
    #     "predict",
    #     f"model={MODEL_PATH}",
    #     f"source={input_path}",
    #     f"device={DEVICE}",       # "0" for GPU, "cpu" otherwise
    #     "save=True",
    #     f"project={project_dir}",
    #     f"name={run_name}",
    #     "exist_ok=True",
    # ]
    cmd = [
    "yolo",
    "predict",
    f"model={MODEL_PATH}",
    f"source={input_path}",
    f"device={DEVICE}",     # "0" for GPU, "cpu" otherwise
    f"imgsz={IMG_SIZE}",    # <<--- add this line
    "save=True",
    f"project={project_dir}",
    f"name={run_name}",
        "exist_ok=True",
    ]

    print("[INFO] Running CLI (image):", " ".join(cmd))
    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("[ERROR] 'yolo' command not found. Make sure Ultralytics CLI is installed.")
        return None

    t1 = time.time()
    print("[INFO] Image CLI finished in {:.2f}s".format(t1 - t0))
    print("[CLI OUTPUT - IMAGE]")
    print(completed.stdout)

    # Find the output image
    save_dir = project_dir / run_name
    if not save_dir.exists():
        print("[ERROR] Expected output dir does not exist:", save_dir)
        return None

    # Look for an image file (jpg/png/webp/bmp)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    images = []
    for ext in exts:
        images.extend(save_dir.glob(f"*{ext}"))

    if not images:
        print("[ERROR] No output image found in", save_dir)
        return None

    out_path = images[0]
    print("[INFO] Output image:", out_path)

    # Copy to a temp file so Gradio can access it safely
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=out_path.suffix)
    tmp_path = Path(tmp.name)
    tmp.close()
    shutil.copy2(out_path, tmp_path)

    return str(tmp_path)


# -----------------------------
# Video prediction (CLI -> GPU) - unchanged, just returns path
# -----------------------------
# def predict_video(video_file):
#     """
#     Use the Ultralytics CLI in a subprocess for video, forcing device=DEVICE.
#     Returns the path to the processed video for Gradio to display.
#     """
#     if video_file is None:
#         return None

#     # Gradio may pass a dict or a plain path string
#     if isinstance(video_file, dict):
#         input_path = video_file.get("name") or video_file.get("path")
#     else:
#         input_path = video_file

#     if input_path is None:
#         print("No video path from Gradio input.")
#         return None

#     input_path = str(input_path)
#     if not os.path.isfile(input_path):
#         print("Could not find video file:", input_path)
#         return None

#     # Ensure a predictable project folder
#     project_dir = Path("gradio_runs")
#     project_dir.mkdir(parents=True, exist_ok=True)

#     # Unique run name
#     run_name = f"gradio_vid_{int(time.time())}"

#     # Build CLI command
#     # cmd = [
#     #     "yolo",
#     #     "predict",
#     #     f"model={MODEL_PATH}",
#     #     f"source={input_path}",
#     #     f"device={DEVICE}",       # "0" for GPU, "cpu" otherwise
#     #     "save=True",
#     #     f"project={project_dir}",
#     #     f"name={run_name}",
#     #     "exist_ok=True",
#     # ]
    
#     cmd = [
#         "yolo",
#         "predict",
#         f"model={MODEL_PATH}",
#         f"source={input_path}",
#         f"device={DEVICE}",     # "0" for GPU, "cpu" otherwise
#         f"imgsz={IMG_SIZE}",    # <<--- add this line
#         "save=True",
#         f"project={project_dir}",
#         f"name={run_name}",
#         "exist_ok=True",
#     ]

#     print("[INFO] Running CLI (video):", " ".join(cmd))
#     t0 = time.time()
#     try:
#         completed = subprocess.run(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             check=False,
#         )
#     except FileNotFoundError:
#         print("[ERROR] 'yolo' command not found. Make sure Ultralytics CLI is installed.")
#         return None

#     t1 = time.time()
#     print("[INFO] Video CLI finished in {:.2f}s".format(t1 - t0))
#     print("[CLI OUTPUT - VIDEO]")
#     print(completed.stdout)

#     # Find the output video (support .mp4, .avi, .mov, .mkv)
#     save_dir = project_dir / run_name
#     if not save_dir.exists():
#         print("[ERROR] Expected output dir does not exist:", save_dir)
#         return None

#     exts = [".mp4", ".avi", ".mov", ".mkv"]
#     vids = []
#     for ext in exts:
#         vids.extend(save_dir.glob(f"*{ext}"))

#     if not vids:
#         print("[ERROR] No video output found in", save_dir)
#         return None

#     out_path = vids[0]
#     print("[INFO] Output video:", out_path)

#     # Keep original extension when copying to temp
#     # tmp = tempfile.NamedTemporaryFile(delete=False, suffix=out_path.suffix)
#     # tmp_path = Path(tmp.name)
#     # tmp.close()
#     # shutil.copy2(out_path, tmp_path)
#     print("[INFO] Output video:", out_path)

#     # If it's already mp4, just copy and return
#     if out_path.suffix.lower() == ".mp4":
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tmp_path = Path(tmp.name)
#         tmp.close()
#         shutil.copy2(out_path, tmp_path)
#         return str(tmp_path)

#     # If it's avi, remux to mp4 with ffmpeg (very fast, -c copy)
#     if out_path.suffix.lower() == ".avi":
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         mp4_path = Path(tmp.name)
#         tmp.close()

#         ffmpeg_cmd = [
#             "ffmpeg",
#             "-y",              # overwrite
#             "-i", str(out_path),
#             "-c", "copy",      # no re-encode → fast
#             str(mp4_path),
#         ]
#         print("[INFO] Remuxing AVI to MP4:", " ".join(ffmpeg_cmd))
#         subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         print("[INFO] Returning mp4:", mp4_path)
#         return str(mp4_path)

#     # Fallback: copy whatever it is
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=out_path.suffix)
#     tmp_path = Path(tmp.name)
#     tmp.close()
#     shutil.copy2(out_path, tmp_path)

#     return str(tmp_path)

import os
import shutil
from pathlib import Path
import tempfile
import time
import subprocess

IMG_SIZE = 384  # or whatever you’re using

FFMPEG_BIN = shutil.which("ffmpeg")  # should now be non-None
if FFMPEG_BIN is None:
    print("[WARN] ffmpeg not on PATH – will return AVI directly.")


def predict_video(video_file):
    """
    Use the Ultralytics CLI in a subprocess for video, forcing device=DEVICE.
    Returns the path (or dict) to the processed video for Gradio to display.
    """
    if video_file is None:
        return None

    # Gradio may pass a dict or a plain path string
    if isinstance(video_file, dict):
        input_path = video_file.get("name") or video_file.get("path")
    else:
        input_path = video_file

    if input_path is None:
        print("No video path from Gradio input.")
        return None

    input_path = str(input_path)
    if not os.path.isfile(input_path):
        print("Could not find video file:", input_path)
        return None

    # Ensure a predictable project folder
    project_dir = Path("gradio_runs")
    project_dir.mkdir(parents=True, exist_ok=True)

    # Unique run name
    run_name = f"gradio_vid_{int(time.time())}"

    cmd = [
        "yolo",
        "predict",
        f"model={MODEL_PATH}",
        f"source={input_path}",
        f"device={DEVICE}",     # "0" for GPU, "cpu" otherwise
        f"imgsz={IMG_SIZE}",
        "save=True",
        f"project={project_dir}",
        f"name={run_name}",
        "exist_ok=True",
    ]

    print("[INFO] Running CLI (video):", " ".join(cmd))
    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("[ERROR] 'yolo' command not found. Make sure Ultralytics CLI is installed.")
        return None

    t1 = time.time()
    print("[INFO] Video CLI finished in {:.2f}s".format(t1 - t0))
    print("[CLI OUTPUT - VIDEO]")
    print(completed.stdout)

    # Find the output video (support .mp4, .avi, .mov, .mkv)
    save_dir = project_dir / run_name
    if not save_dir.exists():
        print("[ERROR] Expected output dir does not exist:", save_dir)
        return None

    exts = [".mp4", ".avi", ".mov", ".mkv"]
    vids = []
    for ext in exts:
        vids.extend(save_dir.glob(f"*{ext}"))

    if not vids:
        print("[ERROR] No video output found in", save_dir)
        return None

    out_path = vids[0]
    print("[INFO] Output video:", out_path)

    # If ffmpeg is available, remux to mp4; otherwise just return the original file
    if FFMPEG_BIN:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_mp4 = Path(tmp.name)
        tmp.close()

        ffmpeg_cmd = [
            FFMPEG_BIN,
            "-y",
            "-i", str(out_path),
            "-c", "copy",
            str(tmp_mp4),
        ]
        print("[INFO] Remuxing AVI to MP4:", " ".join(ffmpeg_cmd))
        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            print("[INFO] Returning mp4:", tmp_mp4)
            # 🔑 Normalize path & return as file + mime_type
            return {
                "name": tmp_mp4.as_posix(),
                "mime_type": "video/mp4",
            }
        except Exception as e:
            print("[WARN] ffmpeg remux failed, returning original video:", e)
            return out_path.as_posix()
    else:
        # No ffmpeg -> just return whatever Ultralytics wrote
        return out_path.as_posix()


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🐶 Dog Breed YOLO Detector")
    gr.Markdown(
        "Upload an **image** or an **MP4 video**.\n\n"
    )

    # ---- Image tab ----
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                # Use filepath so we can pass directly to CLI
                image_input = gr.Image(type="filepath", label="Upload an image")
                image_button = gr.Button("Run detection (image)")
            with gr.Column():
                # Output is also a filepath returned by predict_image
                image_output = gr.Image(type="filepath", label="Detected image")

        image_button.click(fn=predict_image, inputs=image_input, outputs=image_output)

    # ---- Video tab ----
    with gr.Tab("Video (MP4)"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload a video (MP4)")
                video_button = gr.Button("Run video detection")
            with gr.Column():
                video_output = gr.Video(label="Processed video", format="mp4")

        video_button.click(fn=predict_video, inputs=video_input, outputs=video_output)


if __name__ == "__main__":
    demo.launch()

