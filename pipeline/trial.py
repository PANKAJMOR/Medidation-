import os
import sys
import json
import time
import shutil
import requests
from datetime import timedelta
from dotenv import load_dotenv

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.video_ingestion import VideoIngestion
from pipeline.analyze_video import analyze_video

# --- 1. CONFIGURATION ---
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
AUTH_TOKEN = os.getenv("AUTH_TOKEN").strip().replace("\n", "").replace("\r", "")
VIDEO_BASE_DIR = os.path.normpath(os.getenv("VIDEO_STORAGE_DIR"))
PDF_BASE_DIR = os.path.normpath(os.getenv("PDF_REPORT_DIR"))

# Ensure base folders exist
os.makedirs(VIDEO_BASE_DIR, exist_ok=True)
os.makedirs(PDF_BASE_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def process_session(session_data):
    """
    Handles the full workflow for a specific session
    """
    youtube_url = session_data.get("youtubeLink")
    session_id = session_data.get("_id")
    participant_ids = session_data.get("participantsId", [])

    print(f"\n🚀 Processing Session: {session_id}")
    print(f"🔗 Link: {youtube_url}")
    
    video_path = None
    try:
        # --- 2. INGESTION (Saves video in videos/[session_id]/) ---
        print("▶ Step 1: Ingesting Video with Session ID...")
        ingestor = VideoIngestion(base_dir=VIDEO_BASE_DIR)
        
        # We pass session_id to ensure the folder is named correctly
        path, _, mode = ingestor.ingest(youtube_url, video_id=session_id)
        
        if not path:
            print(f"❌ Error: Ingestion failed for session {session_id}")
            return

        video_path = os.path.abspath(os.path.normpath(path))
        
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found at: {video_path}")
            return
            
        print(f"✅ Video saved at: {video_path}")

        # --- 3. ANALYSIS (Passes session_id and participant_ids) ---
        print("▶ Step 2: Running AI Analysis & Mapping PDFs...")
        # analyze_video will now create output/[session_id]/ and name PDFs by participant_ids
        results = analyze_video(video_path, session_id, participant_ids)
        
        if results["status"] == "FAILED":
            print(f"❌ Analysis failed for {session_id}:")
            for error in results.get('errors', []):
                print(f"   - {error['code']}: {error['message']}")
            return

        # --- 4. SUMMARY ---
        print(f"\n✅ Session {session_id} Successful!")
        print(f"📊 Participants Processed: {len(results.get('participants', {}))}")
        
        for p_id, pdf_path in results.get("pdf_reports", {}).items():
            print(f"📄 PDF Generated for {p_id}: {os.path.normpath(pdf_path)}")

    except Exception as e:
        print(f"❌ Unexpected Error in session {session_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup logic (optional: keeps video for review)
        if video_path and os.path.exists(video_path):
            session_folder = os.path.dirname(video_path)
            print(f"🧹 Session folder kept at: {os.path.normpath(session_folder)}")

def get_job_and_process():
    """
    Polls the API for queued jobs and triggers processing
    """
    url = f"{API_BASE_URL}/proctoringTool/queuedJobs"
    print(f"📡 Polling API: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            sessions = data.get("sessions", [])
            
            queued_jobs = [s for s in sessions if s.get("status") == "queued"]
            
            if queued_jobs:
                print(f"🔔 Found {len(queued_jobs)} queued job(s).")
                # Process the first job in the queue
                process_session(queued_jobs[0])
            else:
                print("ℹ️ No queued jobs found.")
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    get_job_and_process()