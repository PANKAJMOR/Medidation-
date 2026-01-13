import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# Add project root for imports
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.video_ingestion import VideoIngestion
from pipeline.analyze_video import analyze_video

# --- 1. CONFIGURATION ---
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip().replace("\n", "").replace("\r", "")
VIDEO_BASE_DIR = os.path.normpath(os.getenv("VIDEO_STORAGE_DIR"))
PDF_BASE_DIR = os.path.normpath(os.getenv("PDF_REPORT_DIR"))

# Ensure base folders exist
os.makedirs(VIDEO_BASE_DIR, exist_ok=True)
os.makedirs(PDF_BASE_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

def process_session(session_data):
    """
    Workflow: Ingest -> Analyze -> Upload Reports -> Prepare Final Payload
    """
    youtube_url = session_data.get("youtubeLink")
    session_id = session_data.get("_id")
    participant_ids = session_data.get("participantsId", [])

    print(f"--- Processing Session: {session_id} ---")
    
    video_path = None
    try:
        # --- 2. INGESTION (Saves in videos/[session_id]/) ---
        print("Step 1: Ingesting Video...")
        ingestor = VideoIngestion(base_dir=VIDEO_BASE_DIR)
        
        # Pass session_id to ensure video folder matches API ID
        path, _, _ = ingestor.ingest(youtube_url, video_id=session_id)
        
        if not path:
            print("Error: Ingestion failed.")
            return

        video_path = os.path.abspath(os.path.normpath(path))
        print(f"Video saved at: {video_path}")

        # --- 3. ANALYSIS ---
        print("Step 2: Running AI Analysis...")
        # analyze_video returns results including pdf_reports mapping
        results = analyze_video(video_path, session_id, participant_ids)
        
        if results["status"] == "FAILED":
            print("Analysis failed:")
            for error in results.get('errors', []):
                print(f"  - {error['code']}: {error['message']}")
            return

        # --- 4. UPLOAD REPORTS (POST /uploadReport) ---
        print("Step 3: Uploading PDF Reports to get Links...")
        final_results_payload = []
        
        # pdf_reports is { "ParticipantID": "Full/Path/To/PDF" }
        for p_id, pdf_path in results.get("pdf_reports", {}).items():
            if os.path.exists(pdf_path):
                print(f"  Uploading report for participant: {p_id}")
                
                # Open as binary to send as BLOB via multipart/form-data
                with open(pdf_path, "rb") as f:
                    upload_data = {"participantId": p_id}
                    upload_files = {"Report": f} # Key must be 'Report'
                    
                    up_resp = requests.post(
                        f"{API_BASE_URL}/proctoringTool/uploadReport",
                        headers=HEADERS,
                        data=upload_data,
                        files=upload_files
                    )
                
                if up_resp.status_code in [200, 201]:
                    # Extract the reportLink from the upload response
                    report_link = up_resp.json().get("reportLink")
                    
                    # Get status from analysis results
                    p_status = results["participants"][p_id]["overall_status"]
                    
                    # Store data for the final payload
                    final_results_payload.append({
                        "participantId": p_id,
                        "reportLink": report_link,
                        "status": p_status
                    })
                    print(f"  Successfully got link: {report_link}")
                else:
                    print(f"  Failed to upload report for {p_id}: {up_resp.text}")

        # --- 5. PRINT FINAL PAYLOAD (FOR VERIFICATION) ---
        # This matches the structure required for /yog/proctoringResult
        final_body = {
            "sessionId": session_id,
            "results": final_results_payload
        }

        print("\n" + "="*50)
        print("DEBUG: PREPARED PAYLOAD FOR /yog/proctoringResult")
        print("="*50)
        print(json.dumps(final_body, indent=4))
        print("="*50 + "\n")

    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if video_path and os.path.exists(video_path):
            session_folder = os.path.dirname(video_path)
            print(f"Session data folder kept at: {os.path.normpath(session_folder)}")

def get_job_and_process():
    """
    Step A: Call the GET API to find queued sessions
    """
    url = f"{API_BASE_URL}/proctoringTool/queuedJobs"
    print(f"Polling API for jobs: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            sessions = data.get("sessions", [])
            
            # Find the first job that is 'queued'
            queued_jobs = [s for s in sessions if s.get("status") == "queued"]
            
            if queued_jobs:
                print(f"Found {len(queued_jobs)} job(s). Starting first job...")
                process_session(queued_jobs[0])
            else:
                print("No queued jobs found.")
        else:
            print(f"API Error {response.status_code}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    get_job_and_process()