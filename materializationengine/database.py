from contextlib import contextmanager
from urllib.parse import urlparse

from dynamicannotationdb import DynamicAnnotationInterface
from flask import current_app
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

from materializationengine.celery_worker import celery_logger
from materializationengine.utils import get_config_param


def get_sql_url_params(sql_url):
    if not isinstance(sql_url, str):
        sql_url = str(sql_url)
    result = urlparse(sql_url)
    url_mapping = {
        "user": result.username,
        "password": result.password,
        "dbname": result.path[1:],
        "host": result.hostname,
        "port": result.port,
    }
    return url_mapping


def reflect_tables(sql_base, database_name):
    sql_uri = f"{sql_base}/{database_name}"
    engine = create_engine(sql_uri)
    meta = MetaData(engine)
    meta.reflect(views=True)
    tables = [table for table in meta.tables]
    engine.dispose()
    return tables


def ping_connection(session):
    is_database_working = True
    try:
        # to check database we will execute raw query
        session.execute("SELECT 1")
    except Exception as e:
        celery_logger.warning(e)
        is_database_working = False
    return is_database_working


class DatabaseConnectionManager:
    """Manages database connections and session lifecycle."""
    
    def __init__(self):
        self._engines = {}
        self._session_factories = {}
        
    def get_engine(self, database_name: str):
        """Get or create SQLAlchemy engine with proper pooling configuration."""
        if database_name not in self._engines:
            SQL_URI_CONFIG = current_app.config["SQLALCHEMY_DATABASE_URI"]
            sql_base_uri = SQL_URI_CONFIG.rpartition("/")[0]
            # Local dev override: a single consolidated database (e.g. matng's
            # test_db) stands in for every aligned_volume/version database name
            # that would otherwise be resolved. Unset in all real configs.
            effective_database_name = current_app.config.get("DATABASE_NAME_OVERRIDE") or database_name
            sql_uri = f"{sql_base_uri}/{effective_database_name}"

            pool_size = current_app.config.get("DB_CONNECTION_POOL_SIZE", 20)
            max_overflow = current_app.config.get("DB_CONNECTION_MAX_OVERFLOW", 30)

            engine = create_engine(
                sql_uri,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections after 30 minutes
                pool_pre_ping=True,  # Test connections before use; reconnect if stale
            )
            # Cache immediately — pool_pre_ping handles connectivity on first checkout.
            # Previously we ran a SELECT 1 test here and didn't cache on failure, which
            # caused a transient error to permanently break this worker's engine cache,
            # leading to repeated ConnectionError → _on_connection_error → consumer isolation.
            self._engines[database_name] = engine
            celery_logger.info(
                f"Created engine for {database_name} "
                f"(pool_size={pool_size}, max_overflow={max_overflow})"
            )

        return self._engines[database_name]
    
    def get_session_factory(self, database_name: str):
        """Get or create scoped session factory for a database."""
        if database_name not in self._session_factories:
            engine = self.get_engine(database_name)
            self._session_factories[database_name] = scoped_session(
                sessionmaker(
                    bind=engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=True  # Ensure objects are not bound to session after commit
                )
            )
        return self._session_factories[database_name]

    @contextmanager
    def session_scope(self, database_name: str):
        """Context manager for database sessions.
        
        Handles proper session lifecycle including commits and rollbacks.
        """
        session_factory = self.get_session_factory(database_name)
        session = session_factory()
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def cleanup(self):
        """Cleanup any remaining sessions and dispose of engine pools."""
        for database_name, session_factory in list(self._session_factories.items()):
            try:
                session_factory.remove()
            except Exception as e:
                celery_logger.error(f"Error cleaning up sessions for {database_name}: {e}")
        
        for database_name, engine in list(self._engines.items()):
            try:
                engine.dispose()
            except Exception as e:
                celery_logger.error(f"Error disposing engine for {database_name}: {e}")
                
        self._session_factories.clear()
        self._engines.clear()
    
    def log_pool_status(self, database_name: str):
        """Log current connection pool status."""
        if database_name in self._engines:
            engine = self._engines[database_name]
            pool = engine.pool
            celery_logger.info(
                f"Pool status for {database_name}: "
                f"checked in={pool.checkedin()}, "
                f"checked out={pool.checkedout()}, "
                f"size={pool.size()}, "
                f"overflow={pool.overflow()}"
            )

class DynamicMaterializationCache:
    def __init__(self):
        self._clients = {}

    def get_db(self, database: str) -> DynamicAnnotationInterface:
        if database not in self._clients:
            db_client = self._get_mat_client(database)
        db_client = self._clients[database]

        connection_ok = ping_connection(db_client.database.cached_session)

        if not connection_ok:
            db_client = self._get_mat_client(database)

        return self._clients[database]

    def _get_mat_client(self, database: str):
        sql_uri_config = get_config_param("SQLALCHEMY_DATABASE_URI")
        pool_size = current_app.config.get("DB_CONNECTION_POOL_SIZE", 20)
        max_overflow = current_app.config.get("DB_CONNECTION_MAX_OVERFLOW", 30)
        # See DatabaseConnectionManager.get_engine's matching override.
        effective_database = get_config_param("DATABASE_NAME_OVERRIDE") or database
        mat_client = DynamicAnnotationInterface(
            sql_uri_config, effective_database, pool_size, max_overflow
        )
        self._clients[database] = mat_client
        return self._clients[database]

    def invalidate_cache(self):
        self._clients = {}


dynamic_annotation_cache = DynamicMaterializationCache()
db_manager = DatabaseConnectionManager()
