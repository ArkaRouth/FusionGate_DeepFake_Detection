import os
import cv2
import torch
import subprocess
import numpy as np
import gradio as gr
from torchvision import models, transforms
from syncnet_python import SyncNetPipeline

# ==========================================
# 1. INITIALIZATION 
# ==========================================
def initialize_models():
    print("Booting up server... Verifying model weights...")
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on device: {compute_device}")
    
    os.makedirs("./weights", exist_ok=True)
    sync_path = "./weights/syncnet_v2.model"
    sfd_path = "./weights/sfd_face.pth"

    if not os.path.exists(sync_path):
        print("Downloading authentic SyncNet weights...")
        torch.hub.download_url_to_file("https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model", sync_path)

    if not os.path.exists(sfd_path):
        print("Downloading authentic Face Detection weights...")
        torch.hub.download_url_to_file("https://huggingface.co/ndkhanh95/LatentSync/resolve/main/checkpoints/auxiliary/sfd_face.pth", sfd_path)

    resnet50 = models.resnet50(weights='IMAGENET1K_V1').eval().to(compute_device)
    pipeline = SyncNetPipeline(
        s3fd_weights=sfd_path,
        syncnet_weights=sync_path,
        device=compute_device
    )
    
    return resnet50, pipeline, compute_device

resnet50, pipeline, compute_device = initialize_models()

preprocess = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. INFERENCE ENGINE
# ==========================================
def predict_video(user_video_path):
    if user_video_path is None:
        return "Please upload a video."
    
    temp_laundered = "./temp_degraded.mp4" 
    
    try:
        # A. Degradation (SAFE FFmpeg command for silent videos)
        cmd = f'ffmpeg -i "{user_video_path}" -vf "gblur=sigma=3" -c:v libx264 -crf 35 -map 0:v -map 0:a? -c:a copy "{temp_laundered}" -y -loglevel error'
        subprocess.run(cmd, shell=True)

        # B. Spatial Score (ResNet)
        cap = cv2.VideoCapture(temp_laundered)
        ret, frame = cap.read()
        cap.release()
        
        s_score = 0
        alpha = 0.5
        
        if ret and frame is not None:
            with torch.no_grad():
                s_score = torch.max(resnet50(preprocess(frame).unsqueeze(0).to(compute_device))).item()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            alpha = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0.1, 0.9)

        # C. Temporal Score (SyncNet)
        try:
            res = pipeline.inference(video_path=temp_laundered)
            t_score = res[1][0] if (len(res) > 6 and res[6]) else 0
        except:
            t_score = 0

        # D. The Fusion Gate
        fused = (alpha * s_score) + ((1 - alpha) * t_score)
        
        # ==========================================
        # 🚨 PLUG IN YOUR GRAPH THRESHOLD HERE 🚨
        # ==========================================
        my_threshold = 3.5  # Change this number based on your Colab graph!
        
        if fused > my_threshold:
            verdict = "🟢 LIKELY REAL"
        else:
            verdict = "🔴 LIKELY FAKE"
        
        return f"Verdict: {verdict}\n\nDeepfake Risk Score: {fused:.2f}\n(Spatial: {s_score:.2f} | Temporal: {t_score:.2f} | Quality: {alpha:.2f})"
        
    except Exception as e:
        return f"Error processing video. Details: {str(e)}"

# ==========================================
# 3. USER INTERFACE
# ==========================================
interface = gr.Interface(
    fn=predict_video, 
    inputs=gr.Video(label="Upload Target Video"),
    outputs=gr.Textbox(label="Analysis Results"),
    title="Fusion Gate Deepfake Detector",
    description="Upload a video to analyze it for spatial and temporal deepfake artifacts."
)

if __name__ == "__main__":
    interface.launch()