import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reporting.pdf_generator import generate_participant_pdf

# --------------------------------------------------
# Test configuration
# --------------------------------------------------
OUTPUT_DIR = "output/reports_test"
PARTICIPANT_ID = "person_0"
ROLE = "CENTER"
INDEX = 0

# --------------------------------------------------
# Mock participant report (matches ReportBuilder output)
# --------------------------------------------------
participant_report = {
    "overall_status": "PASS",
    "neck": {
        "count": 2,
        "allowed": 5,
        "status": "PASS",
        "timestamps": [
            {"start": "10:05:12", "end": "10:05:14"},
            {"start": "10:42:01", "end": "10:42:03"}
        ]
    },
    "arm": {
        "count": 1,
        "allowed": 5,
        "status": "PASS",
        "timestamps": [
            {"start": "11:12:20", "end": "11:12:22"}
        ]
    },
    "leg": {
        "count": 0,
        "allowed": 5,
        "status": "PASS",
        "timestamps": []
    },
    "remarks": []
}

# --------------------------------------------------
# Run test
# --------------------------------------------------
print("Testing PDF generation...\n")

pdf_path = generate_participant_pdf(
    output_dir=OUTPUT_DIR,
    participant_id=PARTICIPANT_ID,
    participant_report=participant_report,
    role=ROLE,
    index=INDEX
)

# --------------------------------------------------
# Verify output
# --------------------------------------------------
if os.path.exists(pdf_path):
    print("PDF generated successfully:")
    print(pdf_path)
else:
    print("PDF generation failed!")
