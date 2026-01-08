import sys
import os
from dotenv import load_dotenv  # ✅ ADD THIS
load_dotenv()  # ✅ ADD THIS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import cv2
import uuid

from prechecks.precheck_manager import PrecheckManager
from prechecks.video_access import VideoAccessCheck
from prechecks.illumination import IlluminationCheck
from prechecks.video_metadata import VideoMetadataCheck
from prechecks.timestamp_check import TimestampCheck
from prechecks.freeze_detection import FreezeCheck
from prechecks.participant_check import ParticipantCheck


from audio.audio_marker import AudioMarker
from movement.movement_manager import MovementManager
from reporting.report_builder import ReportBuilder
# from runtime_checks.freeze_monitor import RuntimeFreezeMonitor
# from runtime_checks.participant_discontinuity import ParticipantDiscontinuity
from yolo.inference import YOLOPoseDetector
from tracking.iou_tracker import IOUTracker
from identity.role_assigner import RoleAssigner
from reporting.pdf_generator import generate_participant_pdf
from reporting.timestamp_converter import convert_movement_timestamps
from runtime_checks.freeze_monitor import RuntimeFreezeMonitor
from runtime_checks.participant_discontinuity import ParticipantDiscontinuity


def analyze_video(video_path, session_id, participant_ids):
    """
    Main production entrypoint
    """
    # 🔑 HARD GUARANTEE
    video_path = os.path.abspath(video_path)

    assert not video_path.startswith("http"), \
        "analyze_video must receive a local file path only"
    
    print("ANALYZE INPUT:", video_path, type(video_path))

    # --------------------------------------------------
    # 1. PRECHECKS (HARD FAILS)
    # --------------------------------------------------
    checks = [
        VideoAccessCheck(),
        IlluminationCheck(),
        VideoMetadataCheck(min_duration_sec=2.75 * 3600),
        TimestampCheck(required_year=2026),
        FreezeCheck(sample_frames=15 * 60),
        ParticipantCheck()
    ]

    prechecks = PrecheckManager(checks)
    precheck_result = prechecks.run_all(video_path)

    if not precheck_result["passed"]:
        return {
            "status": "FAILED",
            "errors": precheck_result["errors"]
        }

    # --------------------------------------------------
    # 2. AUDIO WINDOW DETECTION
    # --------------------------------------------------
    audio_marker = AudioMarker()
    try:
        start_sec, end_sec = audio_marker.get_analysis_window(
            video_path,
            start_ref=r"D:\Meditation proctor\reference_audio\start_audio.wav",
            end_ref=r"D:\Meditation proctor\reference_audio\end_audio.wav"
        )
    except Exception as e:
        return {
            "status": "FAILED",
            "errors": [{
                "code": "AUDIO_MARKER_ERROR",
                "message": str(e)
            }]
        }

    # --------------------------------------------------
    # 3. INITIALIZE PIPELINE COMPONENTS
    # --------------------------------------------------
    detector = YOLOPoseDetector(
        weights="yolov8n-pose.pt",
        conf=0.6,
        imgsz=640
    )

    tracker = IOUTracker(iou_thresh=0.3)
    role_assigner = RoleAssigner()
    movement_manager = MovementManager(fps=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    FRAME_STRIDE = int(fps)  # 1 FPS processing
    frame_idx = 0

    freeze_monitor = RuntimeFreezeMonitor(
        freeze_seconds=15 * 60,  # 15 minutes
        fps=1                   # because FRAME_STRIDE = fps → 1 FPS
    )

    participant_monitor = ParticipantDiscontinuity(
        max_absent_seconds=15,   # configurable
        fps=1                    # because FRAME_STRIDE = fps → 1 FPS
    )


    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    

    # --------------------------------------------------
    # 4. FRAME LOOP (STRICTLY INSIDE AUDIO WINDOW)
    # --------------------------------------------------
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # 🔴 Runtime freeze detection
        freeze_error = freeze_monitor.update(frame)
        if freeze_error:
            cap.release()
            return {
                "status": "FAILED",
                "errors": [freeze_error]
            }


        frame_idx += 1

        # ⛔ Skip before analysis window
        if frame_idx < start_frame:
            continue

        # ⛔ Stop after window
        if frame_idx > end_frame:
            break

        # ⛔ FPS downsampling
        if frame_idx % FRAME_STRIDE != 0:
            continue

        if frame_idx % (FRAME_STRIDE * 60) == 0:
            print(f"Processed {(frame_idx - start_frame) // FRAME_STRIDE} seconds...")

        detections = detector.detect(frame)

        # -------------------------------
        # Prepare bounding boxes
        # -------------------------------
        bboxes = []
        pose_map = {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)

            if area < 6000:
                continue

            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)
            pose_map[tuple(bbox)] = det.keypoints

        bboxes = sorted(
            bboxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )[:3]

        tracked = tracker.update(bboxes)

        # -------------------------------
        # ROLE ASSIGNMENT (ONCE ONLY)
        # -------------------------------
        tracked_people = [(f"person_{tid}", bbox) for tid, bbox in tracked]

        if not role_assigner.assigned and tracked_people:
            role_assigner.assign(tracked_people)

        if role_assigner.assigned and not movement_manager.initialized:
            for pid in role_assigner.role_map:
                movement_manager.register_person(pid)
            movement_manager.initialized = True



        # -------------------------------
        # MOVEMENT PROCESSING
        # -------------------------------
        for track_id, bbox in tracked:
            person_id = f"person_{track_id}"
            keypoints = pose_map.get(tuple(bbox))

            if keypoints is None:
                continue

            x1, y1, x2, y2 = bbox
            face_y2 = y1 + int(0.4 * (y2 - y1))
            face_bbox = (x1, y1, x2, face_y2)

            video_timestamp_sec = (frame_idx - start_frame) / fps

            movement_manager.update(
                person_id=person_id,
                frame=frame,
                keypoints=keypoints,
                face_bbox=face_bbox,
                frame_sec=video_timestamp_sec
            )

            #This participant is currently visible at this second.
            participant_monitor.update(
                person_id=person_id,
                current_sec=video_timestamp_sec
            )               


        # -------------------------------
        # PARTICIPANT QUIT Marker
        # -------------------------------
        discontinued = participant_monitor.check(video_timestamp_sec)

        if discontinued:
            # DO NOT stop analysis — mark and continue
            for pid in discontinued:
                start = participant_monitor.active_absence.get(pid)
                if start is not None:
                    movement_manager.add_discontinuity(
                        pid,
                        start=start,
                        end=video_timestamp_sec
                    )


    cap.release()

    # --------------------------------------------------
    # 5. TIMESTAMP CONVERSION (CRITICAL STEP)
    # --------------------------------------------------
    movement_manager.finalize(
        end_frame_sec=(end_frame - start_frame) / fps
    )
    
    raw_timestamps = movement_manager.get_timestamps()
   

    formatted_timestamps = convert_movement_timestamps(
        raw_timestamps,
        base_offset_sec=start_sec
    )


    # --------------------------------------------------
    # 6. BUILD FINAL REPORT
    # --------------------------------------------------
    report = ReportBuilder()

    final_report = report.build(
        movement_counts=movement_manager.get_all_counts(),
        movement_timestamps=formatted_timestamps
    )

    
    # --------------------------------------------------
    # 7. PDF GENERATION (DYNAMIC PATHS & NAMING)
    # --------------------------------------------------
    pdf_reports = {}
    
    # Get the base directory from .env
    base_output_dir = os.getenv("PDF_REPORT_DIR", "output/pdf_reports")
    
    # Create a subfolder named after the Session ID
    session_specific_dir = os.path.join(base_output_dir, session_id)
    os.makedirs(session_specific_dir, exist_ok=True)
    
    # Sort detected internal IDs (person_0, person_1, etc.) 
    # to ensure consistent mapping to the API participant list
    sorted_person_ids = sorted(final_report.keys())

    for i, person_id in enumerate(sorted_person_ids):
        person_report = final_report[person_id]
        role = role_assigner.role_map.get(person_id, "UNKNOWN")
        index = role_assigner.index_map.get(person_id, -1)

        # Map the internal person_id to the actual database ID by index
        # If there are more people detected than IDs provided, fallback to internal ID
        actual_filename_id = participant_ids[i] if i < len(participant_ids) else person_id

        # Generate the PDF inside the session subfolder with the participant ID as name
        pdf_path = generate_participant_pdf(
            output_dir=session_specific_dir,
            participant_id=actual_filename_id, 
            participant_report=person_report,
            role=role,
            index=index
        )

        # Store the mapping for the return value
        pdf_reports[actual_filename_id] = pdf_path

    return {
        "status": "SUCCESS",
        "participants": final_report,
        "role_mapping": role_assigner.role_map,
        "index_mapping": role_assigner.index_map,
        "pdf_reports": pdf_reports
    }
    
    # # --------------------------------------------------
    # # 7. PDF GENERATION (ONE PER PARTICIPANT)
    # # --------------------------------------------------
    # pdf_reports = {}
    # output_dir = os.getenv("PDF_REPORT_DIR", "output/pdf_reports")
    

    # for person_id, person_report in final_report.items():
    #     role = role_assigner.role_map.get(person_id, "UNKNOWN")
    #     index = role_assigner.index_map.get(person_id, -1)

    #     pdf_path = generate_participant_pdf(
    #         output_dir=output_dir,
    #         participant_id=person_id,
    #         participant_report=person_report,
    #         role=role,
    #         index=index
    #     )

    #     pdf_reports[person_id] = pdf_path


    # return {
    #     "status": "SUCCESS",
    #     "participants": final_report,
    #     "role_mapping": role_assigner.role_map,
    #     "index_mapping": role_assigner.index_map,
    #     "pdf_reports": pdf_reports
    # }

