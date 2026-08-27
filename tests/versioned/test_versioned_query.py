"""
Exercises the consolidated-schema read path (versioned_query.py) against
brain_and_nerve_cord data migrated by hand with matng into a local Postgres
(tests/versioned/conftest.py points every database lookup at it), while
letting validate_datastack/get_relevant_datastack_info keep resolving real
datastack info from the live CAVE info service.

Needs network access to the CAVE global server and a CAVE auth token
(DAF_CREDENTIALS env var, same as production) to resolve
brain_and_nerve_cord's aligned_volume/pcg names and validate its
AnalysisVersion rows. Skips itself if no token is configured.
"""

import pytest
from flask import current_app

from materializationengine.blueprints.client.datastack import validate_datastack
from materializationengine.blueprints.client.versioned_query import query_versioned_table
from materializationengine.database import dynamic_annotation_cache
from materializationengine.info_client import get_relevant_datastack_info

DATASTACK = "brain_and_nerve_cord"
SEGMENTATION_SOURCE = "wclee_fly_cns_001"

# A known-good version present in matng's migrated analysisversion table
# (status=AVAILABLE) -- see matng's docs/running.md for how it was cut.
KNOWN_AVAILABLE_VERSION = 626


@pytest.fixture(autouse=True)
def _require_cave_credentials(test_app):
    if not current_app.config.get("AUTH_TOKEN"):
        pytest.skip(
            "No CAVE auth token configured (DAF_CREDENTIALS); "
            "validate_datastack needs one to reach the live info service."
        )


@validate_datastack
def _resolve_target(datastack_name, version, target_datastack=None, target_version=None):
    # Mirrors the signature validate_datastack expects of a real view function
    # (see views.py's version_view) but does nothing else -- this isolates
    # the decorator's own behavior (info-service lookup + AnalysisVersion
    # validation against the redirected database) from any actual route.
    return target_datastack, target_version


class TestValidateDatastackAgainstRedirectedDb:
    def test_resolves_latest_version(self):
        target_datastack, target_version = _resolve_target(DATASTACK, None)
        assert target_datastack == DATASTACK

    def test_resolves_known_available_version(self):
        target_datastack, target_version = _resolve_target(DATASTACK, KNOWN_AVAILABLE_VERSION)
        assert target_datastack == DATASTACK
        assert target_version == KNOWN_AVAILABLE_VERSION

    def test_missing_version_aborts_404(self):
        from werkzeug.exceptions import NotFound

        with pytest.raises(NotFound):
            _resolve_target(DATASTACK, 999_999_999)


class TestQueryVersionedTable:
    @pytest.fixture(scope="class")
    def db(self):
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(DATASTACK)
        assert pcg_table_name == SEGMENTATION_SOURCE
        return dynamic_annotation_cache.get_db(aligned_volume_name)

    def test_reads_current_rows(self, db):
        df = query_versioned_table(
            db,
            table_name="neck_connective_y92500",
            schema_type="bound_tag",
            segmentation_source=SEGMENTATION_SOURCE,
            limit=10,
        )
        assert len(df) > 0
        assert (df["valid_to_version"] == 9223372036854775807).all()

    def test_reads_as_of_older_version(self, db):
        df = query_versioned_table(
            db,
            table_name="neck_connective_y92500",
            schema_type="bound_tag",
            segmentation_source=SEGMENTATION_SOURCE,
            version=100,
            limit=10,
        )
        assert (df["valid_from_version"] <= 100).all()
        assert (df["valid_to_version"] > 100).all()

    def test_equal_filter_across_tables(self, db):
        current = query_versioned_table(
            db,
            table_name="neck_connective_y92500",
            schema_type="bound_tag",
            segmentation_source=SEGMENTATION_SOURCE,
            limit=1,
        )
        root_id = int(current.iloc[0]["pt_root_id"])

        filtered = query_versioned_table(
            db,
            table_name="backbone_proofread",
            schema_type="proofreading_boolstatus_user",
            segmentation_source=SEGMENTATION_SOURCE,
            equal_filters={"pt_root_id": root_id},
        )
        assert (filtered["pt_root_id"] == root_id).all()
