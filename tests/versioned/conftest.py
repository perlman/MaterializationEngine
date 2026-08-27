"""
Fixtures for testing the consolidated-schema read path (versioned_query.py)
against data migrated by hand with matng, instead of the synthetic
PostGIS-in-docker database the rest of tests/ builds from scratch.

Two things are deliberately different from tests/conftest.py here:

- The autouse `setup_postgis_database` fixture is overridden to a no-op:
  we're reading tables matng already migrated into a real database, not
  creating/populating a fresh one.
- `db_manager`/`dynamic_annotation_cache` are monkeypatched so that *every*
  database name they're asked to connect to resolves to matng's single
  hand-migrated database instead. `validate_datastack` and
  `get_relevant_datastack_info` are untouched -- they still resolve real
  datastack info (aligned_volume name, pcg table name, AnalysisVersion rows)
  from the live CAVE info service; only the physical Postgres connection
  those lookups end up using is redirected.
"""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from dynamicannotationdb import DynamicAnnotationInterface

from materializationengine import database as database_module

TEST_DB_HOST = os.environ.get("MATNG_TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.environ.get("MATNG_TEST_DB_PORT", "5432"))
TEST_DB_USER = os.environ.get("MATNG_TEST_DB_USER", "postgres")
TEST_DB_NAME = os.environ.get("MATNG_TEST_DB_NAME", "test_db")

# matng and MaterializationEngine are assumed to be sibling checkouts (as they
# are set up for local dev); override MATNG_DB_PASSWORD_FILE if that's not the
# case in a given environment.
_DEFAULT_PASSWORD_FILE = Path(__file__).resolve().parents[2].parent / "matng" / "secrets" / "db-local.txt"


def _resolve_test_db_password() -> str:
    explicit = os.environ.get("MATNG_TEST_DB_PASSWORD")
    if explicit:
        return explicit
    password_file = Path(os.environ.get("MATNG_DB_PASSWORD_FILE", _DEFAULT_PASSWORD_FILE))
    if not password_file.exists():
        pytest.skip(
            f"matng test_db password not found at {password_file}; set "
            "MATNG_TEST_DB_PASSWORD or MATNG_DB_PASSWORD_FILE to run tests/versioned."
        )
    return password_file.read_text().strip()


@pytest.fixture(scope="session")
def matng_test_db_uri() -> str:
    password = _resolve_test_db_password()
    return f"postgresql://{TEST_DB_USER}:{password}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"


@pytest.fixture(scope="session", autouse=True)
def setup_postgis_database():
    """Overrides tests/conftest.py's autouse fixture of the same name: this
    test package reads matng's hand-migrated data and must not create or
    populate synthetic tables."""
    yield True


@pytest.fixture(scope="session", autouse=True)
def redirect_db_connections_to_matng_test_db(matng_test_db_uri):
    real_get_engine = database_module.DatabaseConnectionManager.get_engine
    real_get_mat_client = database_module.DynamicMaterializationCache._get_mat_client

    shared_engine = create_engine(
        matng_test_db_uri, poolclass=QueuePool, pool_pre_ping=True,
    )
    shared_interface = DynamicAnnotationInterface(matng_test_db_uri, TEST_DB_NAME)

    def _get_engine(self, database_name: str):
        return shared_engine

    def _get_mat_client(self, database: str):
        self._clients[database] = shared_interface
        return self._clients[database]

    database_module.DatabaseConnectionManager.get_engine = _get_engine
    database_module.DynamicMaterializationCache._get_mat_client = _get_mat_client
    try:
        yield
    finally:
        database_module.DatabaseConnectionManager.get_engine = real_get_engine
        database_module.DynamicMaterializationCache._get_mat_client = real_get_mat_client
        shared_engine.dispose()
