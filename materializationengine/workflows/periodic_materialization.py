"""
Create frozen dataset.
"""
import json
import os
from typing import List

from celery.utils.log import get_task_logger
from materializationengine.blueprints.materialize.api import get_datastack_info
from materializationengine.celery_init import celery
from materializationengine.database import db_manager
from dynamicannotationdb.models import AnalysisVersion
from materializationengine.task import REDIS_CLIENT, argument_signature
from materializationengine.utils import get_config_param
from materializationengine.workflows.complete_workflow import run_complete_workflow

celery_logger = get_task_logger(__name__)


def update_database_workflow_locked(datastack: str, datastack_info: dict) -> bool:
    """Return True if an update_database_workflow for this datastack is already
    queued or running.

    This checks the Redis lock that ``LockedTask`` sets at *enqueue* time (see
    ``materializationengine.task``), not live worker state. That matters under
    KEDA scale-to-zero: the lock is present the moment the workflow is queued,
    so we detect it even while the worker pod is still cold-starting and before
    any task shows up in ``celery.control.inspect().active()``.

    Note: this reconstructs the lock id for the *periodic* enqueue path
    (``run_periodic_database_update``, which passes ``kwargs={"Datastack": ...}``).
    An update triggered manually via the admin API mutates ``datastack_info``
    before enqueuing and therefore locks under a different id; that path is not
    detected here.
    """
    from materializationengine.workflows.update_database_workflow import (
        update_database_workflow,
    )

    lock_id = argument_signature(
        update_database_workflow.name, [datastack_info], {"Datastack": datastack}
    )
    return REDIS_CLIENT.get(lock_id) is not None


def process_datastack(datastack, datastack_info, days_to_expire, merge_tables):
    celery_logger.info(f"Start periodic materialization job for {datastack}")

    aligned_volume = datastack_info["aligned_volume"]["name"]

    with db_manager.session_scope(aligned_volume) as session:
        max_databases = get_config_param("MAX_DATABASES")

        valid_databases = (
            session.query(AnalysisVersion)
            .filter(AnalysisVersion.valid == True)
            .filter(AnalysisVersion.datastack == datastack)
            .filter(AnalysisVersion.parent_version == None)
            .order_by(AnalysisVersion.time_stamp)
            .count()
        )
        if valid_databases >= max_databases:
            celery_logger.info("Number of valid materialized databases is {valid_databases}, threshold is set to: {max_databases}")
            return False
    datastack_info["database_expires"] = True
    task = run_complete_workflow.s(
        datastack_info, days_to_expire=days_to_expire, merge_tables=merge_tables
    )
    task.apply_async(kwargs={"Datastack": datastack})
    return True


@celery.task(name="orchestration:run_periodic_materialization")
def run_periodic_materialization(
    days_to_expire: int = None, merge_tables: bool = True, datastack: str = None
) -> None:
    """
    Run complete materialization workflow. Steps are as follows:
    1. Find missing segmentation data in a given datastack and lookup.
    2. Update expired root ids
    3. Copy database to new frozen version
    4. Merge annotation and segmentation tables together
    5. Drop non-materialized tables
    """
    if datastack:
        datastacks = [datastack]

    else:
        try:
            datastacks = json.loads(os.environ["DATASTACKS"])
        except Exception as e:
            datastacks = get_config_param("DATASTACKS")

    for datastack in datastacks:
        try:
            datastack_info = get_datastack_info(datastack)
            if update_database_workflow_locked(datastack, datastack_info):
                celery_logger.error(
                    f"Update roots workflow is queued or running for {datastack}; "
                    "delaying materialization until it completes."
                )
                continue
            is_running = process_datastack(
                datastack, datastack_info, days_to_expire, merge_tables
            )
            if not is_running:
                celery_logger.error(f"Materialization workflow for {datastack} is not running: {is_running}")
        except Exception as e:
            celery_logger.error(e)
            raise e
