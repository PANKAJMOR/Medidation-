import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

class AudioMarker:
    def __init__(self, min_duration_sec=10600):
        """
        min_duration_sec: minimum analysis duration in seconds (default 2.9 hours)
        """
        self.min_duration_sec = min_duration_sec

    # -------------------------------
    # Extract audio from video
    # -------------------------------
    def extract_audio(self, video_path, audio_path=None):
        if audio_path is None:
            audio_path = os.path.join(os.path.dirname(video_path), "video_audio.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ]
        subprocess.run(cmd, check=True)
        return audio_path

    # -------------------------------
    # Detect timestamp of reference audio
    # -------------------------------
    def detect_audio_timestamp(self, video_audio_path, reference_audio_path, threshold=0.85):
        sr_vid, vid_audio = wavfile.read(video_audio_path)
        sr_ref, ref_audio = wavfile.read(reference_audio_path)

        # Normalize
        vid_audio = vid_audio / np.max(np.abs(vid_audio))
        ref_audio = ref_audio / np.max(np.abs(ref_audio))

        # Cross-correlation
        correlation = correlate(vid_audio, ref_audio, mode='valid')
        correlation /= np.max(correlation)

        peaks = np.where(correlation > threshold)[0]
        if len(peaks) == 0:
            return None

        timestamp_sec = peaks[0] / sr_vid
        return timestamp_sec

    # -------------------------------
    # Get start/end timestamps
    # -------------------------------
    def get_analysis_window(self, video_path, start_ref, end_ref):
        """
        Returns start_sec, end_sec for analysis
        """
        video_audio = self.extract_audio(video_path)

        start_sec = self.detect_audio_timestamp(video_audio, start_ref)
        end_sec = self.detect_audio_timestamp(video_audio, end_ref)

        if start_sec is None:
            raise ValueError("Start audio not detected.")

        # Ignore end audio if it occurs before min_duration
        if end_sec is None or (end_sec - start_sec) < self.min_duration_sec:
            end_sec = start_sec + self.min_duration_sec

        return start_sec, end_sec

    # -------------------------------
    # Slice frames based on timestamps
    # -------------------------------
    def slice_frames(self, frames_dir, start_sec, end_sec, fps=1):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        return frame_files[start_frame:end_frame]

#+++++++++++++++++++++++++version 2+++++++++++++++++++
# import os
# import subprocess
# import numpy as np
# from scipy.io import wavfile
# from scipy.signal import correlate

# class AudioMarker:
#     def __init__(self, fallback_offset_sec=600):
#         """
#         fallback_offset_sec: subtract this many seconds from total duration 
#                              if end reference is not reliable.
#         """
#         self.fallback_offset_sec = fallback_offset_sec

#     # ------------------------------------------------
#     # Extract audio from video
#     # ------------------------------------------------
#     def extract_audio(self, video_path, audio_path=None):
#         if audio_path is None:
#             audio_path = os.path.join(os.path.dirname(video_path), "video_audio.wav")

#         cmd = [
#             "ffmpeg",
#             "-y",
#             "-i", video_path,
#             "-vn",
#             "-ac", "1",
#             "-ar", "16000",
#             audio_path
#         ]
#         subprocess.run(cmd, check=True)
#         return audio_path

#     # ------------------------------------------------
#     # Get video duration (in seconds)
#     # ------------------------------------------------
#     def get_video_duration(self, video_path):
#         cmd = [
#             "ffprobe",
#             "-v", "error",
#             "-show_entries", "format=duration",
#             "-of", "default=noprint_wrappers=1:nokey=1",
#             video_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         return float(result.stdout.strip())

#     # ------------------------------------------------
#     # Detect timestamp of reference audio
#     # ------------------------------------------------
#     def detect_audio_timestamp(self, video_audio_path, reference_audio_path, threshold=0.85):
#         sr_vid, vid_audio = wavfile.read(video_audio_path)
#         sr_ref, ref_audio = wavfile.read(reference_audio_path)

#         # Normalize
#         vid_audio = vid_audio / np.max(np.abs(vid_audio))
#         ref_audio = ref_audio / np.max(np.abs(ref_audio))

#         # Cross-correlation
#         correlation = correlate(vid_audio, ref_audio, mode='valid')
#         correlation /= np.max(correlation)

#         peaks = np.where(correlation > threshold)[0]
#         if len(peaks) == 0:
#             return None

#         timestamp_sec = peaks[0] / sr_vid
#         return timestamp_sec

#     # ------------------------------------------------
#     # Decide start and end points
#     # ------------------------------------------------
#     def get_analysis_window(self, video_path, start_ref, end_ref):
#         video_audio = self.extract_audio(video_path)
#         video_duration = self.get_video_duration(video_path)

#         print(f"Total video duration detected: {video_duration:.2f} sec")

#         # Start detection
#         start_sec = self.detect_audio_timestamp(video_audio, start_ref)
#         if start_sec is None:
#             raise ValueError("Start audio not detected.")

#         # End detection
#         end_sec = self.detect_audio_timestamp(video_audio, end_ref)

#         # ------------------------------------------------
#         # Smart fallback logic
#         # ------------------------------------------------
#         if end_sec is None:
#             print("⚠ End audio NOT found. Using fallback end = duration - 600 sec")
#             end_sec = max(start_sec, video_duration - self.fallback_offset_sec)

#         else:
#             # If end is too early, reject it
#             if end_sec < (video_duration * 0.7):  # occurs too early
#                 print("⚠ End audio found too EARLY. Using fallback end = duration - 600 sec")
#                 end_sec = max(start_sec, video_duration - self.fallback_offset_sec)
#             else:
#                 print("✔ Using detected end audio position")

#         return start_sec, end_sec

#     # ------------------------------------------------
#     # Slice frames based on timestamps
#     # ------------------------------------------------
#     def slice_frames(self, frames_dir, start_sec, end_sec, fps=1):
#         start_frame = int(start_sec * fps)
#         end_frame = int(end_sec * fps)

#         frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
#         return frame_files[start_frame:end_frame]
