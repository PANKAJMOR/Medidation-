import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio.audio_marker import AudioMarker


# -------------------------------
# Paths
# -------------------------------
video_path = r"D:\Meditation proctor\data\31722b92\video.mp4"
frames_dir = r"D:\Meditation proctor\data\31722b92\frames"
start_ref = r"D:\Meditation proctor\reference_audio\start_audio.wav"
end_ref = r"D:\Meditation proctor\reference_audio\end_audio.wav"

# -------------------------------
# Initialize and get analysis window
# -------------------------------
am = AudioMarker(min_duration_sec=10600)  # 2.5 hours default

start_sec, end_sec = am.get_analysis_window(video_path, start_ref, end_ref)
print(f"Start at {start_sec:.2f} sec, End at {end_sec:.2f} sec")

# -------------------------------
# Slice frames for analysis
# -------------------------------
frames_to_analyze = am.slice_frames(frames_dir, start_sec, end_sec, fps=1)
print(f"Total frames for analysis: {len(frames_to_analyze)}")


#+++++++++++++++++++++++++version 2+++++++++++++++++++

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from audio.audio_marker import AudioMarker

# # -------------------------------
# # Paths
# # -------------------------------
# video_path = r"D:\Meditation proctor\data\31722b92\video.mp4"
# frames_dir = r"D:\Meditation proctor\data\31722b92\frames"
# start_ref = r"D:\Meditation proctor\reference_audio\start_audio.wav"
# end_ref = r"D:\Meditation proctor\reference_audio\end_audio.wav"

# # -------------------------------
# # Initialize and get analysis window
# # -------------------------------
# # fallback_offset_sec = 600 → use last 10 minutes trimmed
# am = AudioMarker(fallback_offset_sec=00)

# start_sec, end_sec = am.get_analysis_window(video_path, start_ref, end_ref)
# print(f"\n📌 Start detected at: {start_sec:.2f} sec")
# print(f"📌 End detected at:   {end_sec:.2f} sec\n")

# # -------------------------------
# # Slice frames for analysis
# # -------------------------------
# frames_to_analyze = am.slice_frames(frames_dir, start_sec, end_sec, fps=1)
# print(f"📸 Total frames for analysis: {len(frames_to_analyze)}")
