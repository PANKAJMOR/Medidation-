# Client-approved error codes & messages

ERRORS = {

    # -----------------------------
    # ACCESS / LINK ERRORS
    # -----------------------------
    "VIDEO_NOT_ACCESSIBLE": {
        "client_code": "VIDEO_LINK_NOT_OPENING",
        "message": "Video link not opening or not authorized to open."
    },

    # -----------------------------
    # QUALITY ERRORS
    # -----------------------------
    "POOR_VIDEO_QUALITY": {
        "client_code": "POOR_VIDEO_QUALITY",
        "message": "Not able to analyze due to poor video quality or low illumination."
    },

    # -----------------------------
    # DURATION ERRORS
    # -----------------------------
    "VIDEO_TOO_SHORT": {
        "client_code": "VIDEO_LESS_THAN_3_HRS",
        "message": "Video duration between start and end song is less than 2 hours 45 minutes."
    },

    # -----------------------------
    # PARTICIPANT ERRORS
    # -----------------------------
    "PARTICIPANT_DISCONTINUED": {
        "client_code": "PARTICIPANT_DISCONTINUED",
        "message": "Participant discontinued meditation before completion of 2 hours 45 minutes."
    },

    # -----------------------------
    # TIMESTAMP ERRORS
    # -----------------------------
    "INCORRECT_VIDEO_TIMESTAMP": {
        "client_code": "INCORRECT_VIDEO",
        "message": "Video timestamp does not belong to the calendar year 2026."
    },

    # -----------------------------
    # DISCONTINUITY ERRORS
    # -----------------------------
    "VIDEO_DISCONTINUITY": {
        "client_code": "VIDEO_DISCONTINUITY",
        "message": "Video paused or black screen for more than 15 minutes."
    }
}
