import os
import json
import shutil
from dotenv import load_dotenv

import sys
from datetime import timedelta

# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your modules based on your folder structure
from ingestion.video_ingestion import VideoIngestion
from pipeline.analyze_video import analyze_video

# --- 1. LOAD ENVIRONMENT ---
load_dotenv()

# --- SLASHING LOGIC ---
VIDEO_BASE_DIR = os.path.normpath(os.getenv("VIDEO_STORAGE_DIR"))
PDF_OUT_DIR = os.path.normpath(os.getenv("PDF_REPORT_DIR"))

# Ensure folders exist
os.makedirs(VIDEO_BASE_DIR, exist_ok=True)
os.makedirs(PDF_OUT_DIR, exist_ok=True)

def run_trial(youtube_url):
    print(f"🚀 Starting Trial for: {youtube_url}")
    
    video_path = None
    try:
        # --- 2. INGESTION ---
        print("▶ Step 1: Downloading/Ingesting Video...")
        ingestor = VideoIngestion(base_dir=VIDEO_BASE_DIR)
        
        # ingest() returns (path, id, mode)
        path, internal_id, mode = ingestor.ingest(youtube_url)
        
        if not path:
            print("❌ Error: Video ingestion failed.")
            return

        # --- CRITICAL FIX: Normalize the path for Windows ---
        if isinstance(path, str):
            video_path = os.path.abspath(os.path.normpath(path))
        else:
            print("❌ Error: Video path is not a string (might be stream object)")
            return
        
        # Verify the file actually exists before proceeding
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file does not exist at: {video_path}")
            return
            
        print(f"✅ Video saved at: {video_path}")
        print(f"✅ File exists: {os.path.exists(video_path)}")
        print(f"✅ File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")

        # --- 3. ANALYSIS ---
        print("▶ Step 2: Running AI Analysis & PDF Generation...")
        results = analyze_video(video_path)
        
        if results["status"] == "FAILED":
            print(f"❌ Analysis failed:")
            for error in results['errors']:
                print(f"   - {error['code']}: {error['message']}")
            return

        # --- 4. SHOW RESULTS ---
        print("\n--- TRIAL SUMMARY ---")
        print(f"Status: {results['status']}")
        for person_id, data in results["participants"].items():
            print(f"Person: {person_id} | Status: {data['overall_status']}")
            if person_id in results.get("pdf_reports", {}):
                pdf_path = os.path.normpath(results['pdf_reports'][person_id])
                print(f"📄 PDF Saved: {pdf_path}")

    except Exception as e:
        import traceback
        print(f"❌ Unexpected Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    finally:
        # --- 5. CLEANUP ---
        if video_path and os.path.exists(video_path):
            session_folder = os.path.dirname(video_path)
            clean_session_folder = os.path.normpath(session_folder)
            print(f"🧹 Temporary video folder kept for review: {clean_session_folder}")

if __name__ == "__main__":
    TRIAL_LINK = r"D:\Pankaj\Meditation proctor Main\videos\51cda945\video.mp4" 
    run_trial(TRIAL_LINK)