import os
import sys
import json
import time
import shutil
import requests
from datetime import timedelta
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.video_ingestion import VideoIngestion
from pipeline.analyze_video import analyze_video

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
AUTH_TOKEN = os.getenv("AUTH_TOKEN").strip().replace("\n", "").replace("\r", "")
VIDEO_BASE_DIR = os.path.normpath(os.getenv("VIDEO_STORAGE_DIR"))
PDF_OUT_DIR = os.path.normpath(os.getenv("PDF_REPORT_DIR"))

os.makedirs(VIDEO_BASE_DIR, exist_ok=True)
os.makedirs(PDF_OUT_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def run_trial(youtube_url):
    print(f"🚀 Starting Trial for: {youtube_url}")
    video_path = None
    try:
        print("▶ Step 1: Downloading/Ingesting Video...")
        ingestor = VideoIngestion(base_dir=VIDEO_BASE_DIR)
        path, internal_id, mode = ingestor.ingest(youtube_url)
        
        if not path:
            print("❌ Error: Video ingestion failed.")
            return

        if isinstance(path, str):
            video_path = os.path.abspath(os.path.normpath(path))
        else:
            print("❌ Error: Video path is not a string")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file does not exist at: {video_path}")
            return
            
        print(f"✅ Video saved at: {video_path}")

        print("▶ Step 2: Running AI Analysis & PDF Generation...")
        results = analyze_video(video_path)
        
        if results["status"] == "FAILED":
            print(f"❌ Analysis failed:")
            for error in results['errors']:
                print(f"   - {error['code']}: {error['message']}")
            return

        print("\n--- TRIAL SUMMARY ---")
        print(f"Status: {results['status']}")
        for person_id, data in results["participants"].items():
            print(f"Person: {person_id} | Status: {data['overall_status']}")
            if person_id in results.get("pdf_reports", {}):
                pdf_path = os.path.normpath(results['pdf_reports'][person_id])
                print(f"📄 PDF Saved: {pdf_path}")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    
    finally:
        if video_path and os.path.exists(video_path):
            session_folder = os.path.dirname(video_path)
            clean_session_folder = os.path.normpath(session_folder)
            print(f"🧹 Temporary video folder kept for review: {clean_session_folder}")

def get_job_and_process():
    url = f"{API_BASE_URL}/proctoringTool/queuedJobs"
    print(f"📡 Polling API: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            sessions = data.get("sessions", [])
            
            for session in sessions:
                if session.get("status") == "queued":
                    youtube_link = session.get("youtubeLink")
                    print(f"✅ Found Job: {session['_id']} -> {youtube_link}")
                    run_trial(youtube_link)
                    return
            print("ℹ️ No queued jobs found.")
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    get_job_and_process()