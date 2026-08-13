import datetime
import json
import logging
from typing import Any, Dict

import caveclient
import nglui
import numpy as np
import pandas as pd
from celery.result import AsyncResult
from dynamicannotationdb.models import (
    AnalysisTable,
    AnalysisVersion,
    AnalysisView,
    MaterializedMetadata,
    VersionErrorTable,
)
from dynamicannotationdb.schema import DynamicSchemaClient
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from google.cloud import storage
from middle_auth_client import auth_required, auth_requires_permission
from nglui.statebuilder.helpers import make_point_statebuilder, package_state
from redis import StrictRedis
from sqlalchemy import and_, func, or_
from sqlalchemy.sql import text

from materializationengine.blueprints.client.datastack import validate_datastack
from materializationengine.blueprints.client.query_manager import QueryManager
from materializationengine.blueprints.client.schemas import AnalysisViewSchema
from materializationengine.blueprints.reset_auth import reset_auth
from materializationengine.celery_init import celery
from materializationengine.database import db_manager, dynamic_annotation_cache
from materializationengine.blueprints.client.query_manager import QueryManager
from materializationengine.request_db import request_db_session
from materializationengine.blueprints.client.datastack import validate_datastack

from materializationengine.info_client import (
    get_datastack_info,
    get_datastacks,
    get_relevant_datastack_info,
)
from materializationengine.schemas import (
    AnalysisTableSchema,
    AnalysisVersionSchema,
    AnalysisViewSchema,
    VersionErrorTableSchema,
)
from materializationengine.utils import check_read_permission, get_config_param


__version__ = "5.24.0"

views_bp = Blueprint("views", __name__, url_prefix="/materialize/views")


def make_flat_model(db, table: AnalysisTable):
    anno_metadata = db.database.get_table_metadata(table.table_name)
    ref_table = anno_metadata.get("reference_table", None)
    if ref_table:
        table_metadata = {"reference_table": ref_table}
    else:
        table_metadata = None
    Model = db.schema.create_flat_model(
        table_name=table.table_name,
        schema_type=table.schema,
        table_metadata=table_metadata,
    )
    return Model, anno_metadata


@views_bp.before_request
@reset_auth
def before_request():
    pass



@views_bp.route("/")
@views_bp.route("/index")
@auth_required
def index():
    datastacks = get_datastacks()
    datastack_payload = []
    for datastack in datastacks:
        datastack_data = {"name": datastack}
        try:
            datastack_info = get_datastack_info(datastack)
        except Exception as e:
            logging.warning(e)
            datastack_info = None
        aligned_volume_info = datastack_info.get("aligned_volume")

        if aligned_volume_info:
            datastack_data["description"] = aligned_volume_info.get("description")

        datastack_payload.append(datastack_data)
    return render_template(
        "datastacks.html", datastacks=datastack_payload, version=__version__
    )


@views_bp.route("/cronjobs")
@auth_required
def jobs():
    return render_template("jobs.html", jobs=get_jobs(), version=__version__)


def get_jobs():
    return celery.conf.beat_schedule


@views_bp.route("/cronjobs/<job_name>")
@auth_required
def get_job_info(job_name: str):
    job = celery.conf.beat_schedule[job_name]
    c = job["schedule"]
    now = datetime.datetime.utcnow()
    due = c.is_due(now)

    seconds = now.timestamp()

    next_time_to_run = datetime.datetime.fromtimestamp(seconds + due.next).strftime(
        "%A, %B %d, %Y %I:%M:%S"
    )

    job_info = {
        "cron_schema": c,
        "task": job["task"],
        "kwargs": job["kwargs"],
        "next_time_to_run": next_time_to_run,
    }
    return render_template("job.html", job=job_info, version=__version__)


def make_df_with_links_to_version(
    objects, schema, url, col, col_value, df=None, **urlkwargs
):
    if df is None:
        df = pd.DataFrame(data=schema.dump(objects, many=True))
    if urlkwargs is None:
        urlkwargs = {}
    if url is not None:
        df[col] = df.apply(
            lambda x: "<a href='{}'>{}</a>".format(
                url_for(url, version=getattr(x, col_value), **urlkwargs), x[col]
            ),
            axis=1,
        )
    return df


def make_df_with_links_to_id(
    objects, schema, url, col, col_value, df=None, **urlkwargs
):
    if df is None:
        df = pd.DataFrame(data=schema.dump(objects, many=True))
    if urlkwargs is None:
        urlkwargs = {}
    if url is not None:
        df[col] = df.apply(
            lambda x: "<a href='{}'>{}</a>".format(
                url_for(url, id=getattr(x, col_value), **urlkwargs), x[col]
            ),
            axis=1,
        )
    return df


@views_bp.route("/datastack/<datastack_name>")
@auth_requires_permission("view", table_arg="datastack_name")
def datastack_view(datastack_name):
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    with db_manager.session_scope(aligned_volume_name) as session:

        version_query = session.query(AnalysisVersion).filter(
            AnalysisVersion.datastack == datastack_name
        )
        show_all = request.args.get("all", False) is not False
        if not show_all:
            version_query = version_query.filter(AnalysisVersion.valid == True)
        versions = version_query.order_by(AnalysisVersion.version.desc()).all()

        if len(versions) > 0:
            schema = AnalysisVersionSchema(many=True)
            column_order = schema.declared_fields.keys()
            df = pd.DataFrame(data=schema.dump(versions, many=True))

            df = make_df_with_links_to_id(
                objects=versions,
                schema=schema,
                url="views.version_error",
                col="status",
                col_value="version",
                df=df,
                datastack_name=datastack_name,
            )
            df = make_df_with_links_to_version(
                objects=versions,
                schema=schema,
                url="views.version_view",
                col="version",
                col_value="version",
                df=df,
                datastack_name=datastack_name,
            )
            df = df.reindex(columns=column_order)

            classes = ["table table-borderless"]
            with pd.option_context("display.max_colwidth", None):
                output_html = df.to_html(
                    escape=False, classes=classes, index=False, justify="left", border=0
                )
        else:
            output_html = ""

    return render_template(
        "datastack.html",
        datastack=datastack_name,
        table=output_html,
        version=__version__,
    )


@views_bp.route("/datastack/<datastack_name>/version/<int(signed=True):id>/failed")
@auth_requires_permission("view", table_arg="datastack_name")
def version_error(datastack_name: str, id: int):
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)

    with db_manager.session_scope(aligned_volume_name) as session:

        version = (
            session.query(AnalysisVersion).filter(AnalysisVersion.version == id).first()
        )
        error = (
            session.query(VersionErrorTable)
            .filter(VersionErrorTable.analysisversion_id == version.id)
            .first()
        )
        if error:
            schema = VersionErrorTableSchema()
            error_data = schema.dump(error)
            return render_template(
                "version_error.html",
                traceback=json.dumps(error_data["error"]),
                error_type=error_data["exception"],
                datastack=datastack_name,
                version=__version__,
                mat_version=version.version,
            )

        else:
            return render_template(
                "version_error.html",
                error_type=None,
                datastack=datastack_name,
                version=__version__,
                mat_version=version.version,
            )

def make_precomputed_annotation_link(datastack_name, table_name, client):
    auth_disabled = get_config_param("AUTH_DISABLED", False)
    auth_prefix = "" if auth_disabled else "middleauth+"
    
    seg_source = client.info.segmentation_source(format_for="neuroglancer")
    seg_source = seg_source.replace("graphene://https://", "graphene://middleauth+https://")

    annotation_url = url_for(
        "api.Materialization Client2_live_table_precomputed_info",
        datastack_name=datastack_name,
        table_name=table_name,
        _external=True)
    annotation_source = f"precomputed://{auth_prefix}{annotation_url}"
    annotation_source = annotation_source[:-5]

    seg_sources = [seg_source]
    datastack_info = client.info.get_datastack_info()
    skeleton_source = datastack_info.get("skeleton_source", None)
    if skeleton_source:
        seg_sources.append(skeleton_source)

    seg_layer = nglui.statebuilder.SegmentationLayerConfig(
        source=seg_sources, name="seg"
    )
    img_layer = nglui.statebuilder.ImageLayerConfig(
        source=client.info.image_source(), name="img"
    )

    sb = nglui.statebuilder.StateBuilder([img_layer, seg_layer], client=client)
    json = sb.render_state(
        None,
        return_as="dict",
        url_prefix="https://spelunker.cave-explorer.org",
        target_site="mainline",
    )
    json['layers'].append({
        'type': 'annotation',
        'source': annotation_source,
        'annotationColor': '#ffffff',
        'name': table_name
        })

    sb=nglui.statebuilder.StateBuilder(base_state=json)
    
    url = sb.render_state(url_prefix='https://spelunker.cave-explorer.org', return_as="url", target_site="mainline")
    return url
    
def make_seg_prop_ng_link(datastack_name, table_name, version, client, is_view=False):
    auth_disabled = get_config_param("AUTH_DISABLED", False)
    auth_prefix = "" if auth_disabled else "middleauth+"
    
    seg_source = client.info.segmentation_source(format_for="neuroglancer")
    seg_source = seg_source.replace("graphene://https://", "graphene://middleauth+https://")
    
    if is_view:
        seginfo_url = url_for(
            "api.Materialization Client2_mat_view_segment_info",
            datastack_name=datastack_name,
            version=version,
            view_name=table_name,
            _external=True,
        )
    else:
        seginfo_url = url_for(
            "api.Materialization Client2_mat_table_segment_info",
            datastack_name=datastack_name,
            version=version,
            table_name=table_name,
            _external=True,
        )

    seg_info_source = f"precomputed://{auth_prefix}{seginfo_url}"
    # strip off the /info
    seg_info_source = seg_info_source[:-5]

    seg_sources = [seg_source, seg_info_source]
    datastack_info = client.info.get_datastack_info()
    skeleton_source = datastack_info.get("skeleton_source", None)
    if skeleton_source:
        seg_sources.append(skeleton_source)

    seg_layer = nglui.statebuilder.SegmentationLayerConfig(
        source=seg_sources, name="seg"
    )
    img_layer = nglui.statebuilder.ImageLayerConfig(
        source=client.info.image_source(), name="img"
    )
    sb = nglui.statebuilder.StateBuilder([img_layer, seg_layer], client=client)
    url_link = sb.render_state(
        None,
        return_as="url",
        url_prefix="https://spelunker.cave-explorer.org",
        target_site="mainline",
    )
    return url_link



@views_bp.route("/datastack/<datastack_name>/version/<int(signed=True):version>")
@validate_datastack
@auth_requires_permission("view", table_arg="target_datastack")
def version_view(
    datastack_name: str, version: int, target_datastack=None, target_version=None
):
    """View for displaying analysis tables and views for a specific datastack version."""
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)

    with db_manager.session_scope(aligned_volume_name) as session:
        # Get analysis version - single query instead of duplicate
        analysis_version = get_analysis_version(session, target_datastack, target_version)
        
        if not analysis_version:
            abort(404, f"Version {target_version} not found for datastack {target_datastack}")

        # Create tables DataFrame and add links
        tables_df = create_tables_dataframe(session, analysis_version, target_datastack, aligned_volume_name)
        tables_df = add_table_links(
            tables_df, 
            target_datastack, 
            target_version, 
            aligned_volume_name, 
            current_app.config["GLOBAL_SERVER_URL"]
        )

        # Convert tables to HTML
        column_order = AnalysisTableSchema().declared_fields.keys()
        tables_html = dataframe_to_html(tables_df, column_order)

    # Handle materialized views in separate session
    with db_manager.session_scope(f"{datastack_name}__mat{version}") as mat_session:
        views_df = create_views_dataframe(
            mat_session, 
            target_datastack, 
            target_version, 
            current_app.config["GLOBAL_SERVER_URL"]
        )
        
        if len(views_df) > 0:
            views_html = dataframe_to_html(views_df)
        else:
            views_html = "<h4>No views in datastack</h4>"

    return render_template(
        "version.html",
        datastack=target_datastack,
        analysisversion=target_version,
        table=tables_html,
        view_table=views_html,
        version=__version__,
    )
      

@views_bp.route("/datastack/<datastack_name>/table/<int:id>")
@auth_requires_permission("view", table_arg="datastack_name")
def table_view(datastack_name, id: int):
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)

    with db_manager.session_scope(aligned_volume_name) as session:
        table = session.query(AnalysisTable).filter(AnalysisTable.id == id).first()
        if table is None:
            abort(404, "table not found")
        table_name = table.table_name

    with request_db_session(aligned_volume_name) as db:
        check_read_permission(db, table_name)
   
    # mapping = {
    #     "synapse": url_for(
    #         "views.synapse_report", id=id, datastack_name=datastack_name
    #     ),
    #     "cell_type_local": url_for(
    #         "views.cell_type_local_report", id=id, datastack_name=datastack_name
    #     ),
    # }
    # if table.schema in mapping:
    #     return redirect(mapping[table.schema])
    # else:
    return redirect(
        url_for("views.generic_report", datastack_name=datastack_name, id=id)
    )


@views_bp.route("/datastack/<datastack_name>/table/<int:id>/cell_type_local")
@auth_requires_permission("view", table_arg="datastack_name")
def cell_type_local_report(datastack_name, id):
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    
    # First session scope for getting table metadata
    with db_manager.session_scope(aligned_volume_name) as session:
        table = session.query(AnalysisTable).filter(AnalysisTable.id == id).first()
        if not table:
            abort(404, "Table not found")
        
        if table.schema != "cell_type_local":
            abort(504, "this table is not a cell_type_local table")
        
        # Store values we need after session closes
        table_name = table.table_name
        analysis_version = table.analysisversion.version
        analysis_datastack = table.analysisversion.datastack
        
    # Get database interface and check permissions
    with request_db_session(aligned_volume_name) as db:
        check_read_permission(db, table_name)
        
        # Create model
        Model, anno_metadata = make_flat_model(db, table)
    
    # Second session scope for materialized data
    mat_db_name = f"{datastack_name}__mat{analysis_version}"
    with db_manager.session_scope(mat_db_name) as matsession:
        # Get row count
        n_annotations = (
            matsession.query(MaterializedMetadata)
            .filter(MaterializedMetadata.table_name == table_name)
            .first()
            .row_count
        )
        
        # Build cell type query
        cell_type_merge_query = (
            matsession.query(
                Model.cell_type,
                func.count(Model.cell_type).label("num_cells"),
            )
            .group_by(Model.cell_type)
            .order_by(text("num_cells DESC"))
        ).limit(100)
        
        # Execute query and get dataframe
        df = pd.read_sql(
            cell_type_merge_query.statement,
            db_manager.get_engine(mat_db_name),
            coerce_float=False,
        )

    # Render template with results
    classes = ["table table-borderless"]
    return render_template(
        "cell_type_local.html",
        version=__version__,
        schema_name=table.schema,
        n_annotations=n_annotations,
        table_name=table_name,
        dataset=analysis_datastack,
        table=df.to_html(
            escape=False, 
            classes=classes, 
            index=False, 
            justify="left", 
            border=0
        ),
    )


@views_bp.route("/datastack/<datastack_name>/table/<int:id>/synapse")
@auth_requires_permission("view", table_arg="datastack_name")
def synapse_report(datastack_name, id):
    synapse_task = get_synapse_info.s(datastack_name, id)
    task = synapse_task.apply_async()
    return redirect(
        url_for(
            "views.check_if_complete",
            datastack_name=datastack_name,
            id=id,
            task_id=str(task.id),
        )
    )


@views_bp.route("/datastack/<datastack_name>/table/<int:id>/synapse/<task_id>/loading")
@auth_requires_permission("view", table_arg="datastack_name")
def check_if_complete(datastack_name, id, task_id):
    res = AsyncResult(task_id, app=celery)
    if res.ready() is True:
        synapse_data = res.result

        return render_template(
            "synapses.html",
            synapses=synapse_data["synapses"],
            num_autapses=synapse_data["n_autapses"],
            num_no_root=synapse_data["n_no_root"],
            dataset=datastack_name,
            analysisversion=synapse_data["version"],
            version=__version__,
            table_name=synapse_data["table_name"],
            schema_name="synapse",
        )
    else:
        return render_template("loading.html", version=__version__)


@celery.task(name="process:get_synapse_info", acks_late=True, bind=True)
def get_synapse_info(self, datastack_name, id):
    """Get synapse statistics for a table.
    
    Args:
        datastack_name (str): Name of datastack
        id (int): Analysis table ID
        
    Returns:
        dict: Synapse statistics including counts of total synapses,
              autapses, and synapses without root IDs
    """
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    
    with db_manager.session_scope(aligned_volume_name) as session:
        table = session.query(AnalysisTable).filter(AnalysisTable.id == id).first()
        
        if not table:
            raise ValueError(f"Table with ID {id} not found")
            
        if table.schema != "synapse":
            raise ValueError("This table is not a synapse table")
            
        table_name = table.table_name
        version = table.analysisversion.version
            
    schema_client = DynamicSchemaClient()
    AnnoSynapseModel, SegSynapseModel = schema_client.get_split_models(
        table_name, 
        "synapse",
        pcg_table_name
    )
    
    with db_manager.session_scope(aligned_volume_name) as session:
        # Total synapse count
        synapses = session.query(AnnoSynapseModel).count()
        
        # Count autapses (same pre/post root IDs)
        n_autapses = (
            session.query(AnnoSynapseModel)
            .join(SegSynapseModel, AnnoSynapseModel.id == SegSynapseModel.id)
            .filter(SegSynapseModel.pre_pt_root_id == SegSynapseModel.post_pt_root_id)
            .filter(
                and_(
                    SegSynapseModel.pre_pt_root_id != 0,
                    SegSynapseModel.post_pt_root_id != 0,
                )
            )
            .count()
        )
        
        n_no_root = (
            session.query(AnnoSynapseModel)
            .join(SegSynapseModel, AnnoSynapseModel.id == SegSynapseModel.id)
            .filter(
                or_(
                    SegSynapseModel.pre_pt_root_id == 0,
                    SegSynapseModel.post_pt_root_id == 0,
                )
            )
            .count()
        )
    
    return {
        "table_name": table_name,
        "schema": "synapse",
        "synapses": synapses,
        "n_autapses": n_autapses, 
        "n_no_root": n_no_root,
        "version": version
    }

@views_bp.route(
    "/datastack/<datastack_name>/table/<int:id>/generic", methods=("GET", "POST")
)
@auth_requires_permission("view", table_arg="datastack_name")
def generic_report(datastack_name, id):
    """Generate a generic table report with optional Neuroglancer link generation."""
    
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    
    with db_manager.session_scope(aligned_volume_name) as session:
        table = session.query(AnalysisTable).filter(AnalysisTable.id == id).first()
        if table is None:
            abort(404, "this table does not exist")
            
        parent_version_id = table.analysisversion.parent_version
        if parent_version_id is not None:
            parent_version = session.query(AnalysisVersion).get(parent_version_id)
            target_version = datastack_name
            datastack_name = parent_version.datastack
            aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
                datastack_name
            )
            
        table_name = table.table_name
        version = table.analysisversion.version
        schema = table.schema
        is_merged = table.analysisversion.is_merged
        
    with request_db_session(aligned_volume_name) as db:
        check_read_permission(db, table_name)
        
        anno_metadata = db.database.get_table_metadata(table_name)
    mat_db_name = f"{datastack_name}__mat{version}"
    
    with db_manager.session_scope(mat_db_name) as matsession:
        n_annotations = (
            matsession.query(MaterializedMetadata)
            .filter(MaterializedMetadata.table_name == table_name)
            .first()
            .row_count
        )
    
    qm = QueryManager(
        mat_db_name,
        segmentation_source=pcg_table_name,
        meta_db_name=aligned_volume_name,
        split_mode=not is_merged,
        direct_sql_pandas=True
    )
    qm.add_table(table_name)
    qm.select_all_columns(table_name)
    df, column_names = qm.execute_query()
    
    if request.method == "POST":
        pos_column = request.form["position"]
        grp_column = request.form.get("group", None)
        linked_cols = request.form.get("linked", None)
        
        data_res = [
            anno_metadata["voxel_resolution_x"],
            anno_metadata["voxel_resolution_y"],
            anno_metadata["voxel_resolution_z"],
        ]
        
        client = caveclient.CAVEclient(
            datastack_name, 
            server_address=current_app.config["GLOBAL_SERVER_URL"]
        )
        
        sb = make_point_statebuilder(
            client,
            point_column=pos_column,
            group_column=grp_column,
            linked_seg_column=linked_cols,
            data_resolution=data_res,
        )
        
        url = package_state(
            df,
            sb,
            client,
            shorten="always",
            return_as="url",
            ngl_url=None,
            link_text="link",
        )
        return redirect(url)
    
    classes = ["table table-borderless"]
    with pd.option_context("display.max_colwidth", None):
        output_html = df.to_html(
            escape=False,
            classes=classes,
            index=False,
            justify="left",
            border=0,
            table_id="datatable",
        )
    
    root_columns = [c for c in df.columns if c.endswith("_root_id")]
    pt_columns = [c for c in df.columns if c.endswith("_position")]
    sv_id_columns = [c for c in df.columns if c.endswith("_supervoxel_id")]
    id_valid = ["id", "valid"]
    
    all_system_cols = np.concatenate(
        [root_columns, pt_columns, sv_id_columns, id_valid]
    )
    other_columns = df.columns[~df.columns.isin(all_system_cols)]
    
    return render_template(
        "generic.html",
        pt_columns=pt_columns,
        root_columns=root_columns,
        other_columns=other_columns,
        dataset=datastack_name,
        analysisversion=version,
        version=__version__,
        table_name=table_name,
        schema_name=schema,
        n_annotations=n_annotations,
        anno_metadata=anno_metadata,
        table=output_html,
    )

# Constants for version_view
TABLE_CSS_CLASSES = ["table", "table-borderless"]
SCHEMA_URL_TEMPLATE = "<a href='{}/schema/views/type/{}/view'>{}</a>"


def get_analysis_version(session, datastack, version):
    """Get analysis version for a datastack and version number."""
    analysis_version = (
        session.query(AnalysisVersion)
        .filter(AnalysisVersion.version == version)
        .filter(AnalysisVersion.datastack == datastack)
        .first()
    )
    return analysis_version


def create_tables_dataframe(session, analysis_version, target_datastack, aligned_volume_name):
    """Create DataFrame for analysis tables with links."""
    table_query = session.query(AnalysisTable).filter(
        AnalysisTable.analysisversion == analysis_version
    )
    tables = table_query.all()
    
    if not tables:
        return pd.DataFrame()
    
    df = make_df_with_links_to_id(
        objects=tables,
        schema=AnalysisTableSchema(many=True),
        url="views.table_view",
        col="id",
        col_value="id",
        datastack_name=target_datastack,
    )
    
    return df


def add_table_links(df, target_datastack, target_version, aligned_volume_name, global_server_url):
    """Add ng_link and schema links to tables DataFrame."""
    if df.empty:
        return df
        
    client = caveclient.CAVEclient(
        target_datastack, server_address=global_server_url
    )
    
    df["ng_link"] = df.apply(
        lambda x: f"<a href='{make_seg_prop_ng_link(target_datastack, x.table_name, target_version, client)}'>seg prop link</a> \
                    <a href='{make_precomputed_annotation_link(target_datastack, x.table_name, client)}'>annotation link</a>",
        axis=1
    )
    
    df["schema"] = df.schema.map(
        lambda x: SCHEMA_URL_TEMPLATE.format(global_server_url, x, x)
    )
    
    df["table_name"] = df.table_name.map(
        lambda x: f"<a href='/annotation/views/aligned_volume/{aligned_volume_name}/table/{x}'>{x}</a>"
    )
    
    return df


def create_views_dataframe(mat_session, target_datastack, target_version, global_server_url):
    """Create DataFrame for analysis views with links."""
    views = mat_session.query(AnalysisView).all()
    
    if not views:
        return pd.DataFrame()
    
    views_df = make_df_with_links_to_id(
        objects=views,
        schema=AnalysisViewSchema(many=True),
        url=None,
        col=None,
        col_value=None,
        datastack_name=target_datastack,
    )
    
    if len(views_df) > 0:
        client = caveclient.CAVEclient(
            target_datastack, server_address=global_server_url
        )
        views_df["ng_link"] = views_df.apply(
            lambda x: f"<a href='{make_seg_prop_ng_link(target_datastack, x.table_name, target_version, client, is_view=True)}'>seg prop link</a>",
            axis=1,
        )
    
    return views_df


def dataframe_to_html(df, column_order=None):
    """Convert DataFrame to HTML with consistent styling."""
    if df.empty:
        return "<h4>No data available</h4>"
    
    if column_order:
        df = df.reindex(columns=list(column_order) + (["ng_link"] if "ng_link" in df.columns else []))
    
    with pd.option_context("display.max_colwidth", None):
        return df.to_html(
            escape=False, 
            classes=TABLE_CSS_CLASSES, 
            index=False, 
            justify="left", 
            border=0
        )