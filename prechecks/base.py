from abc import ABC, abstractmethod

class PrecheckResult:
    def __init__(self, ok: bool, error_code=None, message=None):
        self.ok = ok
        self.error_code = error_code
        self.message = message


class BasePrecheck(ABC):

    @abstractmethod
    def run(self, video_path) -> PrecheckResult:
        pass
