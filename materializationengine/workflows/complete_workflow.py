import datetime

from celery import chain, chord
from celery.utils.log import get_task_logger
from materializationengine.celery_init import celery
from materializationengine.shared_tasks import (
    fin,
    get_materialization_info,
    workflow_complete,
    monitor_workflow_state,
    workflow_failed,
)
from materializationengine.workflows.create_frozen_database import (
    check_tables,
    create_materialized_database_workflow,
    create_new_version,
    format_materialization_database_workflow,
    rebuild_reference_tables,
    set_version_status,
    clean_split_table_workflow,
)
from materializationengine.workflows.ingest_new_annotations import (
    ingest_new_annotations_workflow,
    find_missing_root_ids_workflow
)
from materializationengine.task import LockedTask, RedisLease
from materializationengine.workflows.update_root_ids import (
    update_root_ids_workflow,
)

celery_logger = get_task_logger(__name__)


@celery.task(
    bind=True,
    name="orchestration:run_complete_workflow",
)
def run_complete_workflow(
    self,
    datastack_info: dict,
    days_to_expire: int = 5,
    merge_tables: bool = True,
    **kwargs,
):
    """Run complete materialization workflow.
    Workflow overview:
        - Find all annotations with missing segmentation rows
        and lookup supervoxel_id and root_id
        - Lookup all expired root_ids and update them
        - Copy the database to a new versioned database
        - Merge annotation and segmentation tables

    Args:
        datastack_info (dict): [description]
        days_to_expire (int, optional): [description]. Defaults to 5.
    """
    # Guard the *execution* against duplicates for this datastack. The task is
    # acks_late=False and blocks in monitor_workflow_state for the whole workflow
    # (hours); broker redeliveries (and any duplicate enqueue) re-run the SAME
    # message on fresh KEDA orchestration pods, so without this every duplicate
    # would create another database version and update root ids concurrently --
    # which corrupts the copy. A heartbeat-refreshed lease keyed by datastack
    # lets one run proceed and turns every concurrent duplicate into a no-op.
    datastack = datastack_info["datastack"]
    lease = RedisLease(f"RUN_COMPLETE_WORKFLOW_LOCK:{datastack}")
    if not lease.acquire():
        celery_logger.warning(
            f"run_complete_workflow for {datastack} is already running "
            f"(lease held by {lease.holder!r}); skipping duplicate execution "
            f"{self.request.id}."
        )
        return False
    try:
        return _run_complete_workflow(
            self, datastack_info, days_to_expire, merge_tables, **kwargs
        )
    finally:
        lease.release()


def _run_complete_workflow(
    self,
    datastack_info: dict,
    days_to_expire: int = 5,
    merge_tables: bool = True,
    **kwargs,
):
    materialization_time_stamp = datetime.datetime.utcnow()

    new_version_number = create_new_version(
        datastack_info=datastack_info,
        materialization_time_stamp=materialization_time_stamp,
        days_to_expire=days_to_expire,
        merge_tables=merge_tables,
    )

    mat_info = get_materialization_info(
        datastack_info, new_version_number, materialization_time_stamp
    )
    celery_logger.info(mat_info)

    update_live_database_workflow = []

    # lookup missing segmentation data for new annotations and update expired root_ids
    # skip tables that are larger than 1,000,000 rows due to performance.
    for mat_metadata in mat_info:
        celery_logger.info(
            f"Running workflow for {mat_metadata['annotation_table_name']}"
        )
        if mat_metadata.get("segmentation_table_name"):
            workflow = chain(
                ingest_new_annotations_workflow(mat_metadata),
                # find_missing_root_ids_workflow(mat_metadata), # skip for now
                update_root_ids_workflow(mat_metadata),
            )
        else:
            workflow = fin.si()

        update_live_database_workflow.append(workflow)

    celery_logger.debug(f"CHAINED TASKS: {update_live_database_workflow}")
    # copy live database as a materialized version and drop unneeded tables
    setup_versioned_database_workflow = create_materialized_database_workflow(
        datastack_info, new_version_number, materialization_time_stamp, mat_info
    )

    # drop indices, merge annotation and segmentation tables and re-add indices on merged table
    if merge_tables:
        format_database_workflow = format_materialization_database_workflow(mat_info)
        analysis_database_workflow = chain(
            chord(format_database_workflow, fin.si()),
            rebuild_reference_tables.si(mat_info),
            check_tables.si(mat_info, new_version_number),
        )
    else:
        clean_split_tables = clean_split_table_workflow(mat_info=mat_info)
        analysis_database_workflow = chain(
            chord(clean_split_tables, fin.si()),
            check_tables.si(mat_info, new_version_number),
        )

    # combine all workflows into final workflow and run
    workflow = chain(
        *update_live_database_workflow,
        setup_versioned_database_workflow,
        analysis_database_workflow,
        set_version_status.si(mat_info, new_version_number, "AVAILABLE"),
        workflow_complete.si("Materialization workflow"),
    )
    final_workflow = workflow.apply_async(
        kwargs={"Datastack": datastack_info["datastack"]},
        link_error=workflow_failed.s(mat_info=mat_info),
    )
    is_complete = monitor_workflow_state(final_workflow)
    celery_logger.info(f"Workflow: {final_workflow} is complete {is_complete}")
