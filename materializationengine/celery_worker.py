import datetime
import logging
import os
import signal
import sys
import threading
import time
import warnings
from typing import Any, Callable, Dict

import redis
from celery.app.builtins import add_backend_cleanup_task
from celery.schedules import crontab
from celery.signals import after_setup_logger, worker_process_init
from celery.utils.log import get_task_logger
from dateutil import relativedelta
from marshmallow import ValidationError

from materializationengine.celery_init import celery
from materializationengine.celery_slack import post_to_slack_on_task_failure
from materializationengine.errors import ConfigurationError
from materializationengine.schemas import CeleryBeatSchema
from materializationengine.utils import get_config_param

celery_logger = get_task_logger(__name__)


def create_celery(app=None):
    celery.conf.broker_url = app.config["CELERY_BROKER_URL"]
    celery.conf.result_backend = app.config["CELERY_RESULT_BACKEND"]
    if app.config.get("USE_SENTINEL", False):
        celery.conf.broker_transport_options = {
            "master_name": app.config["MASTER_NAME"]
        }
        celery.conf.result_backend_transport_options = {
            "master_name": app.config["MASTER_NAME"]
        }
    # Configure Celery and related loggers
    log_level = app.config["LOGGING_LEVEL"]
    celery_logger.setLevel(log_level)
    
    # Configure all Celery internal loggers to suppress noisy messages
    celery_loggers = [
        'celery',
        'celery.worker',
        'celery.worker.consumer', 
        'celery.worker.strategy',
        'celery.worker.heartbeat',
        'celery.worker.job',
        'celery.beat',
        'celery.control',
        'celery.app.trace'
    ]
    
    for logger_name in celery_loggers:
        logging.getLogger(logger_name).setLevel(log_level)
    
    # Debug: Check if BEAT_SCHEDULES is in app.config
    beat_schedules = app.config.get("BEAT_SCHEDULES", [])
    celery_logger.debug(f"BEAT_SCHEDULES from app.config: {beat_schedules}")
    celery_logger.debug(f"BEAT_SCHEDULES type: {type(beat_schedules)}, length: {len(beat_schedules) if isinstance(beat_schedules, (list, dict)) else 'N/A'}")
    
    celery.conf.update(
        {
            "task_routes": ("materializationengine.task_router.TaskRouter"),
            "task_serializer": "json",
            "result_serializer": "json",
            "accept_content": ["json", "application/json"],
            "optimization": "fair",
            "task_send_sent_event": True,
            "task_track_started": True,
            "worker_send_task_events": True,
            "worker_prefetch_multiplier": 1,
            "result_expires": 86400,  # results expire in broker after 1 day
            "redis_socket_connect_timeout": 10,
            "broker_transport_options": {
                "visibility_timeout": 21600,
                "socket_timeout": 20,
                "socket_connect_timeout": 20,
            },  # timeout (s) for tasks to be sent back to broker queue
            "beat_schedules": beat_schedules,
        }
    )

    celery.conf.update(app.config)
    # Ensure beat_schedules is set correctly after update (in case app.config overwrote it)
    # Use BEAT_SCHEDULES from app.config if beat_schedules is empty or missing
    if not celery.conf.get("beat_schedules") and app.config.get("BEAT_SCHEDULES"):
        celery.conf.beat_schedules = app.config["BEAT_SCHEDULES"]
        celery_logger.debug(f"Restored beat_schedules from BEAT_SCHEDULES: {len(app.config['BEAT_SCHEDULES'])} schedules")
    
    # Debug: Verify beat_schedules is in celery.conf after update
    celery_logger.debug(f"beat_schedules in celery.conf after update: {celery.conf.get('beat_schedules', 'NOT FOUND')}")
    celery_logger.debug(f"BEAT_SCHEDULES in celery.conf after update: {celery.conf.get('BEAT_SCHEDULES', 'NOT FOUND')}")
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    if os.environ.get("SLACK_WEBHOOK"):
        celery.Task.on_failure = post_to_slack_on_task_failure

    configure_worker_autoshutdown(app)

    return celery


def configure_worker_autoshutdown(app):
    """Let a worker exit after a bounded amount of work so it behaves correctly
    as a KEDA ScaledJob pod (one pod -> a little work -> exit), instead of
    running forever like a normal long-lived worker.

    Config (written by the helm chart into the orchestration config.cfg; absent
    -> feature disabled, so ordinary workers such as producer/consumer/api are
    unaffected):

      CELERY_WORKER_AUTOSHUTDOWN_ENABLED        master on/off switch
      CELERY_WORKER_AUTOSHUTDOWN_MAX_TASKS      exit after this many completed tasks
      CELERY_WORKER_AUTOSHUTDOWN_DELAY_SECONDS  grace period after the last task before
                                                exiting (lets acks / result writes settle)
      CELERY_WORKER_IDLE_TIMEOUT_SECONDS        if > 0, exit when no task has started for
                                                this long -- covers "started but the queue
                                                was empty". Set small for a near-immediate
                                                exit on an empty queue (0 = disabled).

    Deliberately task-agnostic (counts any task) so it can be reused on any queue.
    """
    if not app.config.get("CELERY_WORKER_AUTOSHUTDOWN_ENABLED", False):
        return

    from celery.signals import task_postrun, task_prerun, worker_ready

    max_tasks = int(app.config.get("CELERY_WORKER_AUTOSHUTDOWN_MAX_TASKS", 1))
    shutdown_delay = int(app.config.get("CELERY_WORKER_AUTOSHUTDOWN_DELAY_SECONDS", 2))
    idle_timeout = int(app.config.get("CELERY_WORKER_IDLE_TIMEOUT_SECONDS", 0))

    state = {"completed": 0, "idle_timer": None, "shutting_down": False}
    lock = threading.Lock()

    def _shutdown(reason, delay):
        with lock:
            if state["shutting_down"]:
                return
            state["shutting_down"] = True
            if state["idle_timer"] is not None:
                state["idle_timer"].cancel()
                state["idle_timer"] = None
        celery_logger.info(f"[autoshutdown] {reason}; exiting in {delay}s")

        def _send_term():
            if delay > 0:
                time.sleep(delay)
            # Warm-shutdown THIS worker only (finish any in-flight task, stop
            # consuming, exit). Do NOT use app.control.shutdown(): that broadcasts
            # to every worker on the queue and would kill sibling pods too.
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_send_term, name="autoshutdown", daemon=True).start()

    def _arm_idle_timer():
        if idle_timeout <= 0:
            return
        with lock:
            if state["shutting_down"]:
                return
            if state["idle_timer"] is not None:
                state["idle_timer"].cancel()
            timer = threading.Timer(
                idle_timeout, _shutdown, args=(f"no task started for {idle_timeout}s", 0)
            )
            timer.daemon = True
            state["idle_timer"] = timer
            timer.start()

    @worker_ready.connect(weak=False)
    def _on_worker_ready(**_kwargs):
        # If the queue was empty at startup (or nothing arrives), exit rather
        # than linger. A task arriving cancels this via task_prerun.
        _arm_idle_timer()

    @task_prerun.connect(weak=False)
    def _on_task_prerun(**_kwargs):
        with lock:
            if state["idle_timer"] is not None:
                state["idle_timer"].cancel()
                state["idle_timer"] = None

    @task_postrun.connect(weak=False)
    def _on_task_postrun(**_kwargs):
        with lock:
            state["completed"] += 1
            completed = state["completed"]
        if completed >= max_tasks:
            _shutdown(f"completed {completed}/{max_tasks} task(s)", shutdown_delay)
        else:
            # More tasks allowed on this pod; wait for the next, but don't linger.
            _arm_idle_timer()


@after_setup_logger.connect
def celery_loggers(logger, *args, **kwargs):
    """
    Add stdout handler for Celery logger output.
    """
    logger.addHandler(logging.StreamHandler(sys.stdout))


@worker_process_init.connect
def configure_http_connection_pools(sender=None, **kwargs):
    """
    Increase GCS urllib3 connection pool sizes for each forked worker process.

    CloudVolume uses cloud-files with use_https=True, which converts gs:// paths to
    https://storage.googleapis.com/... and routes them through HttpInterface.
    HttpInterface holds a class-level HTTPAdapter (created at import time with the
    default pool_maxsize=10) that is shared across ALL instances and sessions.
    With CLOUDVOLUME_PARALLEL concurrent threads all funnelling through that one
    adapter, the pool fills immediately → discarded connections → TCP+TLS handshake
    on every request.

    Three-part fix (all run once per forked worker process):
    1. Patch HTTPAdapter.__init__ so any new Session/AuthorizedSession created after
       this hook uses pool_maxsize=GCS_CONNECTION_POOL_SIZE (default: 128).
    2. Replace HttpInterface.adaptor (the shared class-level adapter) with a fresh
       HTTPAdapter that has the larger pool.  This is the critical fix for use_https
       paths because the old adapter was created before the patch could apply.
    3. Reset cloud-files' GC_POOL and invalidate cloudvolume_cache so any gs://
       connections inherited from the parent process are discarded; fresh ones pick
       up the patched HTTPAdapter.

    Tune with GCS_CONNECTION_POOL_SIZE environment variable (default: 128).
    """
    from requests.adapters import HTTPAdapter
    import cloudfiles.interfaces as cf_interfaces
    from materializationengine.cloudvolume_gateway import cloudvolume_cache

    pool_size = int(os.environ.get("GCS_CONNECTION_POOL_SIZE", "128"))
    _orig_init = HTTPAdapter.__init__

    def _patched_init(self, pool_connections=pool_size, pool_maxsize=pool_size, **kw):
        _orig_init(self, pool_connections=pool_connections, pool_maxsize=pool_maxsize, **kw)

    HTTPAdapter.__init__ = _patched_init

    # Replace the class-level adapter shared by all HttpInterface instances.
    # This is the primary fix for use_https=True (https://storage.googleapis.com)
    # paths: the old class-level adapter has pool_maxsize=10 and cannot be patched
    # retroactively via HTTPAdapter.__init__.
    cf_interfaces.HttpInterface.adaptor = HTTPAdapter(
        pool_connections=pool_size, pool_maxsize=pool_size
    )

    # For gs:// paths (non-use_https): discard GCS bucket connections inherited
    # from the parent process.  reset_connection_pools() replaces the global
    # GC_POOL with fresh empty queues; the next gs:// request creates a new
    # google.cloud.storage.Client → AuthorizedSession → patched HTTPAdapter.
    cf_interfaces.reset_connection_pools()

    # Clear any CloudVolume client objects that hold references to old connections.
    # They are re-populated lazily on first use in this worker process.
    cloudvolume_cache.invalidate_cache()

    celery_logger.info(
        f"[worker_process_init] GCS connection pool reset: "
        f"HTTPAdapter defaults patched to pool_maxsize={pool_size}, "
        f"HttpInterface.adaptor replaced, GC_POOL reset, cloudvolume_cache invalidated."
    )
    

def days_till_next_month(date):
    """function to pick out the same weekday in the next month
    So if you pass the first wednesday of January, you get
    the first wednesday of February

    Args:
        date (datetime.datetime): a timepoint

    Returns:
        datetime.datetime: same day next month (in the sense of same # of weekday)
    """

    weekday = relativedelta.weekday(date.isoweekday() - 1)
    weeknum = (date.day - 1) // 7 + 1
    weeknum = weeknum if weeknum <= 4 else 4
    next_date = date + relativedelta.relativedelta(
        months=1, day=1, weekday=weekday(weeknum)
    )
    delta_days = next_date - date
    return delta_days.days


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # remove expired task results in redis broker
    sender.add_periodic_task(
        crontab(hour=0, minute=0, day_of_week="*", day_of_month="*", month_of_year="*"),
        add_backend_cleanup_task(celery),
        name="Clean up back end results",
    )

    # Try to get beat_schedules from celery.conf, fallback to BEAT_SCHEDULES if not found
    beat_schedules = celery.conf.get("beat_schedules")
    if not beat_schedules:
        # Fallback: try to get from BEAT_SCHEDULES (uppercase) in celery.conf
        beat_schedules = celery.conf.get("BEAT_SCHEDULES", [])
        if beat_schedules:
            celery_logger.debug(f"Found BEAT_SCHEDULES (uppercase), converting to beat_schedules")
            celery.conf.beat_schedules = beat_schedules
    
    celery_logger.debug(f"beat_schedules from celery.conf: {beat_schedules}")
    celery_logger.debug(f"beat_schedules type: {type(beat_schedules)}, length: {len(beat_schedules) if isinstance(beat_schedules, (list, dict)) else 'N/A'}")
    
    if not beat_schedules:
        celery_logger.info("No periodic tasks configured.")
        return
    try:
        schedules = CeleryBeatSchema(many=True).load(beat_schedules)
    except ValidationError as validation_error:
        celery_logger.error(f"Configuration validation failed: {validation_error}")
        raise ConfigurationError("Invalid configuration") from validation_error

    min_databases = sender.conf.get("MIN_DATABASES")
    celery_logger.info(f"MIN_DATABASES: {min_databases}")
    for schedule in schedules:
        try:
            task = configure_task(schedule, min_databases)
            sender.add_periodic_task(
                create_crontab(schedule),
                task,
                name=schedule["name"],
            )
            celery_logger.info(f"Added task: {schedule['name']}")
        except ConfigurationError as e:
            celery_logger.error(
                f"Error configuring task '{schedule.get('name', 'Unknown')}': {str(e)}"
            )


def configure_task(schedule: Dict[str, Any], min_databases: int = None) -> Callable:
    task_name = schedule["task"]
    datastack_params = schedule.get("datastack_params", {})

    if is_old_materialization_configuration(task_name):
        return schedule_legacy_workflow(task_name)
    else:
        return schedule_workflow(task_name, datastack_params, min_databases)


def is_old_materialization_configuration(task_name: str) -> bool:
    old_task_names = [
        "run_daily_periodic_materialization",
        "run_weekly_periodic_materialization",
        "run_lts_periodic_materialization",
    ]
    return task_name in old_task_names


def schedule_legacy_workflow(task_name: str) -> Callable:
    from materializationengine.workflows.periodic_materialization import (
        run_periodic_materialization,
    )

    warnings.warn(
        f"Deprecated task name '{task_name}' detected. Please update your configuration to use 'run_periodic_materialization' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if task_name == "run_daily_periodic_materialization":
        days_to_expire = 2
    elif task_name == "run_weekly_periodic_materialization":
        days_to_expire = 7
    elif task_name == "run_lts_periodic_materialization":
        days_to_expire = days_till_next_month(
            datetime.datetime.now(datetime.timezone.utc)
        )
    else:
        raise ConfigurationError(f"Unknown old task name: {task_name}")

    return run_periodic_materialization.s(
        days_to_expire=days_to_expire,
        merge_tables=False,  # Default value for old configuration
    )


def schedule_workflow(
    task_name: str,
    datastack_params: Dict[str, Any],
    min_databases: int = None,
) -> Callable:
    from materializationengine.workflows.periodic_database_removal import (
        remove_expired_databases,
    )
    from materializationengine.workflows.periodic_materialization import (
        run_periodic_materialization,
    )
    from materializationengine.workflows.update_database_workflow import (
        run_periodic_database_update,
    )

    if task_name == "remove_expired_databases":
        default_threshold_for_task = 5
        if min_databases is not None:
            default_threshold_for_task = min_databases

        delete_threshold = datastack_params.get(
            "delete_threshold", default_threshold_for_task
        )

        return remove_expired_databases.s(
            delete_threshold=delete_threshold,
            datastack=datastack_params.get("datastack"),
        )

    elif task_name == "run_periodic_database_update":
        return run_periodic_database_update.s(
            datastack=datastack_params.get("datastack")
        )

    elif task_name == "run_periodic_materialization":
        return run_periodic_materialization.s(
            days_to_expire=datastack_params.get("days_to_expire", 2),
            merge_tables=datastack_params.get("merge_tables", False),
            datastack=datastack_params.get("datastack"),
        )

    else:
        raise ConfigurationError(f"Unknown task: {task_name}")


def create_crontab(schedule: Dict[str, Any]) -> crontab:
    """Create a crontab object from the schedule dictionary."""
    return crontab(
        minute=schedule.get("minute", "*"),
        hour=schedule.get("hour", "*"),
        day_of_week=schedule.get("day_of_week", "*"),
        day_of_month=schedule.get("day_of_month", "*"),
        month_of_year=schedule.get("month_of_year", "*"),
    )


def get_celery_worker_status():
    i = celery.control.inspect()
    availability = i.ping()
    stats = i.stats()
    registered_tasks = i.registered()
    active_tasks = i.active()
    scheduled_tasks = i.scheduled()
    result = {
        "availability": availability,
        "stats": stats,
        "registered_tasks": registered_tasks,
        "active_tasks": active_tasks,
        "scheduled_tasks": scheduled_tasks,
    }
    return result


def get_celery_queue_items(queue_name: str):
    with celery.connection_or_acquire() as conn:
        return conn.default_channel.queue_declare(
            queue=queue_name, passive=True
        ).message_count


def get_activate_tasks():
    inspector = celery.control.inspect()
    return inspector.active()


def inspect_locked_tasks(release_locks: bool = False):
    client = redis.StrictRedis(
        host=get_config_param("REDIS_HOST"),
        port=get_config_param("REDIS_PORT"),
        password=get_config_param("REDIS_PASSWORD"),
        db=0,
    )

    locked_tasks = list(client.scan_iter(match="LOCKED_WORKFLOW_TASK*"))
    lock_status_dict = {locked_task: {"locked": True} for locked_task in locked_tasks}

    if release_locks:
        for locked_task in lock_status_dict:
            client.delete(locked_task)
            lock_status_dict[locked_task] = {"locked": False}
    return lock_status_dict
