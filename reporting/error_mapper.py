from reporting.error_codes import ERRORS

def map_error(internal_code):
    if internal_code not in ERRORS:
        return {
            "code": "INTERNAL_EXCEPTION",
            "message": str(internal_code)
        }

    e = ERRORS[internal_code]
    return {
        "code": e["client_code"],
        "message": e["message"]
    }
