"""
Minimal read-only query path for the consolidated (single-database,
multi-version) schema produced by matng's migration tools.

Unlike QueryManager (query_manager.py), which serves the existing per-version-
database materialization (joins on annmodel.id == segmodel.id, "live" vs. a
frozen snapshot database), this module targets a single table whose
segmentation model carries valid_from_version/valid_to_version and is joined
on annmodel.id == segmodel.anno_id (see emannotationschemas.models'
with_version_columns and docs/extending_models_for_versioning.md in matng).

Scope is deliberately narrow: single table, equality filters only, no joins
across tables, no spatial filters, no writes. It exists to prototype reads
against tables migrated by hand with matng -- not to replace QueryManager.
"""

import pandas as pd
from dynamicannotationdb import DynamicAnnotationInterface
from sqlalchemy import func, text

# Matches matng's db_common.MAX_BIGINT -- the "still current" sentinel stamped
# into valid_to_version by the target database's current_mat_marker() default.
MAX_BIGINT = 9223372036854775807


def get_versioned_models(
    db: DynamicAnnotationInterface,
    table_name: str,
    schema_type: str,
    segmentation_source: str,
    table_metadata: dict = None,
):
    """Return (anno_model, seg_model) for a table migrated onto the
    consolidated schema. seg_model is None for schema types that carry no
    segmentation fields (nothing for matng to have added version columns to).
    """
    return db.schema.get_split_models(
        table_name=table_name,
        schema_type=schema_type,
        segmentation_source=segmentation_source,
        table_metadata=table_metadata,
        with_version_columns=True,
    )


def is_consolidated_table(
    db: DynamicAnnotationInterface, table_name: str, segmentation_source: str,
) -> bool:
    """Whether table_name's segmentation table has been migrated onto the
    consolidated schema (carries valid_to_version), as opposed to the plain
    1:1 segmentation table shape a production per-version-database
    deployment still uses. Used to route between generic_report (the
    production per-version-database report) and consolidated_report (this
    module's report) at the same table_view redirect point."""
    seg_table_name = f"{table_name}__{segmentation_source}"
    result = db.database.session.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table_name AND column_name = 'valid_to_version';"
        ),
        {"table_name": seg_table_name},
    ).first()
    return result is not None


def _apply_version_filter(query, seg_model, version: int = None):
    if version is None:
        return query.filter(seg_model.valid_to_version == MAX_BIGINT)
    return query.filter(
        seg_model.valid_from_version <= version,
        seg_model.valid_to_version > version,
    )


def count_versioned_table(
    db: DynamicAnnotationInterface,
    table_name: str,
    schema_type: str,
    segmentation_source: str,
    version: int = None,
    table_metadata: dict = None,
) -> int:
    """Row count matching query_versioned_table's same version filter,
    without fetching row data -- cheap even for large tables."""
    anno_model, seg_model = get_versioned_models(
        db, table_name, schema_type, segmentation_source, table_metadata,
    )
    session = db.database.session
    if seg_model is not None:
        query = session.query(func.count(anno_model.id)).join(
            seg_model, anno_model.id == seg_model.anno_id
        )
        query = _apply_version_filter(query, seg_model, version)
    else:
        query = session.query(func.count(anno_model.id))
    return query.scalar()


def query_versioned_table(
    db: DynamicAnnotationInterface,
    table_name: str,
    schema_type: str,
    segmentation_source: str,
    version: int = None,
    equal_filters: dict = None,
    table_metadata: dict = None,
    limit: int = None,
) -> pd.DataFrame:
    """Read rows from a table migrated onto the consolidated schema.

    Parameters
    ----------
    db : DynamicAnnotationInterface
        Already pointed at the target (migrated) database.
    table_name, schema_type, segmentation_source : str
        Same meaning as elsewhere in this codebase -- see
        DynamicSchemaClient.get_split_models.
    version : int, optional
        Materialization version to read as-of: rows with
        valid_from_version <= version < valid_to_version. If omitted, reads
        the current/live rows (valid_to_version == MAX_BIGINT). Ignored for
        schema types with no segmentation model (nothing to version).
    equal_filters : dict, optional
        {column_name: value} equality filters. Looked up on the segmentation
        model first (if present), falling back to the annotation model.
    limit : int, optional
        Row limit.

    Returns
    -------
    pd.DataFrame
        One row per matching annotation. For a segmented table, the leading
        columns are `anno_id` (the stable identity -- constant across every
        version-row of the same annotation) and `seg_id` (the segmentation
        row's own surrogate key -- a *different* value on each version-row,
        since a new segmentation row is inserted whenever the annotation's
        root_id changes; compare it across versions to see that happen). The
        annotation table's own `id` is omitted since it's always identical to
        `anno_id` (that equality is exactly the join condition below).
    """
    anno_model, seg_model = get_versioned_models(
        db, table_name, schema_type, segmentation_source, table_metadata,
    )

    session = db.database.session

    if seg_model is not None:
        anno_data_columns = [c for c in anno_model.__table__.columns if c.key != "id"]
        seg_data_columns = [
            c for c in seg_model.__table__.columns if c.key not in ("id", "anno_id")
        ]
        select_columns = [seg_model.anno_id, seg_model.id, *anno_data_columns, *seg_data_columns]
        column_names = ["anno_id", "seg_id", *(c.key for c in anno_data_columns), *(c.key for c in seg_data_columns)]

        query = session.query(*select_columns).join(seg_model, anno_model.id == seg_model.anno_id)
        query = _apply_version_filter(query, seg_model, version)
    else:
        select_columns = list(anno_model.__table__.columns)
        column_names = [c.key for c in select_columns]
        query = session.query(*select_columns)

    if equal_filters:
        for column_name, value in equal_filters.items():
            model = seg_model if seg_model is not None and hasattr(seg_model, column_name) else anno_model
            query = query.filter(getattr(model, column_name) == value)

    if limit is not None:
        query = query.limit(limit)

    rows = query.all()
    return pd.DataFrame(rows, columns=column_names)
