from reporting.error_mapper import map_error

class PrecheckManager:

    def __init__(self, checks):
        self.checks = checks

    def run_all(self, video_path):
        errors = []

        for check in self.checks:
            result = check.run(video_path)
            if not result.ok:
                errors.append(map_error(result.error_code))

        return {
            "passed": len(errors) == 0,
            "errors": errors
        }
