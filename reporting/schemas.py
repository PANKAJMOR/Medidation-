def empty_person_report():
    return {
        "neck": {
            "count": 0,
            "allowed": 0,
            "status": "PASS"
        },
        "arm": {
            "count": 0,
            "allowed": 0,
            "status": "PASS"
        },
        "leg": {
            "count": 0,
            "allowed": 0,
            "status": "PASS"
        },
        "overall_status": "PASS",
        "remarks": []
    }
