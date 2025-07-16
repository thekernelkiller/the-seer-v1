from sqlalchemy import Engine, create_engine
from typing_extensions import Self

from common.utils.logging import logger


class DatabaseManager:
    _instance = None

    def __new__(cls, dsn: str = "sqlite:///./sample.db") -> Self:
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            logger.info(f"{type(cls._instance).__name__} instance initialized...")
            try:
                cls._instance.client = create_engine(url=dsn, echo=True)
                logger.info(f"Initialized {type(cls._instance.client).__name__} successfully...")
            except Exception as e:
                logger.error(f"Error when initializing {type(cls._instance).__name__}...\n\n{e}")
                cls._instance = None
        return cls._instance

    def __repr__(self) -> str:
        return f"<{type(self._instance).__name__} with client={self._instance.client}>"

    def get_connection(self) -> Engine:
        return self._instance.client


if __name__ == "__main__":
    db = DatabaseManager()
    print(db)