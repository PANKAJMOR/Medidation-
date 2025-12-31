import sys
import os


# --------------------------------------------------
# Add project root
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.run_analysis import run_analysis

INPUT = "https://www.youtube.com/live/Rxo4dN-f-B8?si=5cTcIo97tru"   # or Drive link or local path

result = run_analysis(INPUT)

print(result)
