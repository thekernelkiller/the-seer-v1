import logging


class LogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs) -> None:
        super(LogRecord, self).__init__(*args, **kwargs)
        self.pathname = self.custom_path_filter(self.pathname)

    @staticmethod
    def custom_path_filter(path: str) -> str:
        project_root = "/trademarkia-ai"
        idx = path.find(project_root)
        if idx != -1:
            path = path[idx:]
        return "."+path


logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s] [%(pathname)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()
