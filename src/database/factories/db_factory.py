from ..adapters.postgress_db_adapter import PostgresDbAdapter
from ..base.base_db import BaseDB


class DBFactory:
    @staticmethod
    def get_db(db_type: str, **kwargs) -> BaseDB:
        db_type = db_type.lower()
        if db_type in ["postgres", "pgvector", "postgresql"]:
            return PostgresDbAdapter(**kwargs)
        else:
            raise ValueError(f"Unsupported DB type: {db_type}")
