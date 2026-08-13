import copy
import datetime
import io
import json
from functools import wraps
from typing import List

import cloudvolume
import nglui
import numpy as np
import pandas as pd
import pytz
import werkzeug
from cachetools import LRUCache, TTLCache, cached
from cachetools.keys import hashkey
from dynamicannotationdb.models import AnalysisTable, AnalysisVersion
from emannotationschemas.schemas.base import PostGISField, SegmentationField
from flask import Response, abort, current_app, g, request
from flask_accepts import accepts
from flask_restx import Namespace, Resource, inputs, reqparse
from geoalchemy2.types import Geometry
from marshmallow import fields as mm_fields
from middle_auth_client import (
    auth_requires_permission,
)
from neuroglancer import viewer_state
from sqlalchemy.sql.sqltypes import Boolean, DateTime, Float, Integer, Numeric, String

from materializationengine.blueprints.client.common import (
    generate_complex_query_dataframe,
    generate_simple_query_dataframe,
    get_analysis_version_and_table,
    get_analysis_version_and_tables,
    handle_complex_query,
    handle_simple_query,
    sql_query_warning,
    validate_table_args,
)
from materializationengine.blueprints.client.cache import get_cached_view_metadata
from materializationengine.blueprints.client.common import (
    unhandled_exception as common_unhandled_exception,
)
from materializationengine.blueprints.client.datastack import validate_datastack
from materializationengine.blueprints.client.new_query import (
    remap_query,
    strip_root_id_filters,
    update_rootids,
)
from materializationengine.blueprints.client.precomputed import AnnotationWriter
from materializationengine.blueprints.client.query_manager import QueryManager
from materializationengine.blueprints.client.schemas import (
    AnalysisViewSchema,
    ComplexQuerySchema,
    SimpleQuerySchema,
    V2QuerySchema,
)
from materializationengine.blueprints.client.utils import (
    after_request,
    collect_crud_columns,
    create_query_response,
    get_latest_version,
    update_notice_text_warnings,
)
from materializationengine.blueprints.reset_auth import reset_auth
from materializationengine.chunkedgraph_gateway import chunkedgraph_cache
from materializationengine.database import db_manager, dynamic_annotation_cache
from materializationengine.info_client import get_aligned_volumes, get_datastack_info
from materializationengine.limiter import limit_by_category
from materializationengine.models import MaterializedMetadata
from materializationengine.request_db import request_db_session
from materializationengine.schemas import AnalysisTableSchema, AnalysisVersionSchema
from materializationengine.utils import check_read_permission

__version__ = "5.24.0"


authorizations = {
    "apikey": {"type": "apiKey", "in": "query", "name": "middle_auth_token"}
}

client_bp = Namespace(
    "Materialization Client2",
    authorizations=authorizations,
    description="Materialization Client",
)


@client_bp.errorhandler(werkzeug.exceptions.BadRequest)
def bad_request_exception(e):
    raise e


@client_bp.errorhandler(Exception)
def unhandled_exception(e):
    return common_unhandled_exception(e)


annotation_parser = reqparse.RequestParser()
annotation_parser.add_argument(
    "annotation_ids", type=int, action="split", help="list of annotation ids"
)
annotation_parser.add_argument(
    "pcg_table_name", type=str, help="name of pcg segmentation table"
)


def _get_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{value} is not a valid float")


class float_range(object):
    """Restrict input to an float in a range (inclusive)"""

    def __init__(self, low, high, argument="argument"):
        self.low = low
        self.high = high
        self.argument = argument

    def __call__(self, value):
        value = _get_float(value)
        if value < self.low or value > self.high:
            msg = "Invalid {arg}: {val}. {arg} must be within the range {lo} - {hi}"
            raise ValueError(
                msg.format(arg=self.argument, val=value, lo=self.low, hi=self.high)
            )
        return value

    @property
    def __schema__(self):
        return {
            "type": "integer",
            "minimum": self.low,
            "maximum": self.high,
        }


query_parser = reqparse.RequestParser()
query_parser.add_argument(
    "return_pyarrow",
    type=inputs.boolean,
    default=True,
    required=False,
    location="args",
    help=(
        "whether to return query in pyarrow compatible binary format"
        "(faster), false returns json"
    ),
)
query_parser.add_argument(
    "arrow_format",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help=("whether to convert dataframe to pyarrow ipc batch format"),
)
query_parser.add_argument(
    "random_sample",
    type=inputs.positive,
    default=None,
    required=False,
    location="args",
    help="How many samples to randomly get using tablesample on annotation tables, useful for visualization of large tables does not work as a random sample of query",
)
query_parser.add_argument(
    "split_positions",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help=("whether to return position columns" "as seperate x,y,z columns (faster)"),
)
query_parser.add_argument(
    "count",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to only return the count of a query",
)
query_parser.add_argument(
    "allow_missing_lookups",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to return annotation results when there\
 are new annotations that exist but haven't yet had supervoxel and \
rootId lookups. A warning will still be returned, but no 406 error thrown.",
)
query_parser.add_argument(
    "allow_invalid_root_ids",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to let a query proceed when passed a set of root ids\
 that are not valid at the timestamp that is queried. If True the filter will likely \
not be relevant and the user might not be getting data back that they expect, but it will not error.",
)
query_parser.add_argument(
    "ipc_compress",
    type=inputs.boolean,
    default=True,
    required=False,
    location="args",
    help="whether to have arrow compress the result when using \
          return_pyarrow=True and arrow_format=True. \
          If False, the result will not have it's internal data\
          compressed (note that the entire response \
          will be gzip compressed if accept-enconding includes gzip). \
          If True, accept-encoding will determine what \
          internal compression is used",
)
query_parser.add_argument(
    "direct_sql_pandas",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to use direct SQL queries with pandas, \
          if False it will fall back to the csv streaming method. \
          which is prone to mangling types. \
          CAVEclient>=8.0.0 should set this to True",
)

query_seg_prop_parser = reqparse.RequestParser()
# add an argument for a string controlling the label format
query_seg_prop_parser.add_argument(
    "label_format",
    type=str,
    default=None,
    required=False,
    location="args",
    help="string controlling the label format, should be formatted like a python format string,\
          i.e. {cell_type}_{id}, utilizing the columns available in the response",
)
# add an argument which is a list of column strings
query_seg_prop_parser.add_argument(
    "label_columns",
    type=str,
    action="split",
    default=None,
    required=False,
    location="args",
    help="list of column names include in a label (will be overridden by label_format)",
)
metadata_parser = reqparse.RequestParser()
# add a boolean argument for whether to return all expired versions
metadata_parser.add_argument(
    "expired",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to return all expired versions",
)

def fix_dataframe_types(df):
    """Fix dataframe types for nglui.
    Args:
        df (pd.DataFrame): dataframe to fix types for.
    Returns:
        pd.DataFrame: dataframe with fixed types."""
    for colname in df.columns:
        if df[colname].isnull().all():
            df.drop(columns=[colname], inplace=True)
            continue
        if pd.api.types.is_float_dtype(df[colname]):
            df[colname]=df[colname].astype(np.float32)
        if pd.api.types.is_integer_dtype(df[colname]):
            df[colname]=df[colname].astype(int)
        if pd.api.types.is_string_dtype(df[colname]):
            df[colname]=df[colname].astype(str)
    return df

@cached(cache=TTLCache(maxsize=64, ttl=600))
def get_relevant_datastack_info(datastack_name):
    ds_info = get_datastack_info(datastack_name=datastack_name)
    seg_source = ds_info["segmentation_source"]
    pcg_table_name = seg_source.split("/")[-1]
    aligned_volume_name = ds_info["aligned_volume"]["name"]
    return aligned_volume_name, pcg_table_name


def check_aligned_volume(aligned_volume):
    aligned_volumes = get_aligned_volumes()
    if aligned_volume not in aligned_volumes:
        abort(400, f"aligned volume: {aligned_volume} not valid")


def get_closest_versions(datastack_name: str, timestamp: datetime.datetime):
    avn, _ = get_relevant_datastack_info(datastack_name)
    analysis_version_schema = AnalysisVersionSchema() # Instantiate the schema


    with db_manager.session_scope(avn) as session:
        # query analysis versions to get a valid version which is
        # the closest to the timestamp while still being older
        # than the timestamp
        past_version = (
            session.query(AnalysisVersion)
            .filter(AnalysisVersion.datastack == datastack_name)
            .filter(AnalysisVersion.valid == True)
            .filter(AnalysisVersion.time_stamp < timestamp)
            .order_by(AnalysisVersion.time_stamp.desc())
            .first()
        )

        if past_version:
            past_v_data = analysis_version_schema.dump(past_version)
        else:
            past_v_data = None

        # query analysis versions to get a valid version which is
        # the closest to the timestamp while still being newer
        # than the timestamp
        future_version = (
            session.query(AnalysisVersion)
            .filter(AnalysisVersion.datastack == datastack_name)
            .filter(AnalysisVersion.valid == True)
            .filter(AnalysisVersion.time_stamp > timestamp)
            .order_by(AnalysisVersion.time_stamp.asc())
            .first()
        )

        if future_version:
            future_v_data = analysis_version_schema.dump(future_version)
        else:
            future_v_data = None
    return past_v_data, future_v_data, avn


def check_column_for_root_id(col):
    if isinstance(col, str):
        if col.endswith("root_id"):
            abort(400, "we are not presently supporting joins on root_ids")
    elif isinstance(col, list):
        for c in col:
            if c.endwith("root_id"):
                abort(400, "we are not presently supporting joins on root ids")


def check_joins(joins):
    for join in joins:
        check_column_for_root_id(join[1])
        check_column_for_root_id(join[3])
        
        
def execute_materialized_query(
    datastack: str,
    aligned_volume: str,
    mat_version: int,
    pcg_table_name: str,
    user_data: dict,
    query_map: dict,
    cg_client,
    random_sample: int = None,
    split_mode: bool = False,
    direct_sql_pandas: bool = False,
) -> pd.DataFrame:
    """_summary_

    Args:
        datastack (str): datastack to query on
        mat_version (int): verison to query on
        user_data (dict): dictionary of query payload including filters
        query_map (dict): mapping of model column names to dataframe column names
        cg_client: chunkedgraph client to use for root id lookups
        random_sample (int, optional): number of random samples to get using TABLESAMPLE. Defaults to None.
        split_mode (bool, optional): whether to use split mode for the query. Defaults to False.
        direct_sql_pandas (bool, optional): whether to use pandas for the query. Defaults to False.

    Returns:
        pd.DataFrame: a dataframe with the results of the query in the materialized version
        dict[dict]: a dictionary of table names, with values that are a dictionary
        that has keys of model column names, and values of their name in the dataframe with suffixes added
        if necessary to disambiguate.
    """
    mat_db_name = f"{datastack}__mat{mat_version}"
    with db_manager.session_scope(mat_db_name) as session:    
        mat_row_count = (
            session.query(MaterializedMetadata.row_count)
            .filter(MaterializedMetadata.table_name == user_data["table"])
            .scalar()
        )
        # Validate random_sample to prevent TABLESAMPLE errors
        if random_sample is not None:
            if not np.isfinite(random_sample) or random_sample <= 0:
                print(f"WARNING: Invalid random_sample: {random_sample}, setting to None")
                random_sample = None
            elif mat_row_count <= 0:
                print(f"WARNING: Invalid mat_row_count: {mat_row_count}, setting random_sample to None")
                random_sample = None
            elif random_sample >= mat_row_count:
                random_sample = None
            else:
                percentage = (100.0 * random_sample) / mat_row_count
                if not np.isfinite(percentage) or percentage <= 0:
                    print(f"WARNING: Invalid percentage calculation: {percentage}, setting random_sample to None")
                    random_sample = None
                elif percentage > 100.0:
                    print(f"WARNING: Percentage > 100%: {percentage}, setting random_sample to None")
                    random_sample = None
                else:
                    random_sample = percentage

        if mat_row_count:
            # Decide between TABLESAMPLE and hash-based sampling based on sample size
            hash_config = user_data.get("hash_sampling_config")
            use_hash_sampling = False
            use_random_sample = random_sample
            
            if hash_config and hash_config.get("enabled"):
                # Use QUERY_LIMIT_SIZE from Flask config instead of parameter
                max_points = current_app.config.get("PRECOMPUTED_OVERVIEW_MAX_SIZE", 50000)
                
                # Get configurable threshold for switching from TABLESAMPLE to hash sampling
                hash_sampling_threshold = current_app.config.get("HASH_SAMPLING_THRESHOLD_PERCENT", 5.0)
                volume_fraction = hash_config.get("volume_fraction", 1.0)
                
                # Validate volume_fraction to prevent invalid calculations
                if not np.isfinite(volume_fraction) or volume_fraction <= 0:
                    print(f"WARNING: Invalid volume_fraction in hash_config: {volume_fraction}, using 1.0")
                    volume_fraction = 1.0
                elif volume_fraction > 1.0:
                    print(f"WARNING: volume_fraction > 1.0 in hash_config: {volume_fraction}, capping at 1.0")
                    volume_fraction = 1.0
                    
                # Calculate what percentage of the table we need to sample
                if mat_row_count > 0 and volume_fraction > 0:
                    sample_percentage = (max_points * 100.0) / (mat_row_count * volume_fraction)
                else:
                    print(f"WARNING: Invalid values for percentage calculation: mat_row_count={mat_row_count}, volume_fraction={volume_fraction}")
                    sample_percentage = 100.0  # Fallback to no sampling
                
                if sample_percentage >= 100.0:  # Table is small enough - show all points
                    # No sampling needed, table has fewer rows than QUERY_LIMIT_SIZE
                    use_hash_sampling = False
                    use_random_sample = None
                elif sample_percentage < hash_sampling_threshold:  # Less than threshold - use TABLESAMPLE
                    # Calculate percentage needed (with some buffer to account for randomness)
                    use_random_sample = sample_percentage
                    # Validate that use_random_sample is valid for TABLESAMPLE
                    if not np.isfinite(use_random_sample) or use_random_sample <= 0:
                        print(f"WARNING: Invalid use_random_sample: {use_random_sample}, switching to hash sampling")
                        use_hash_sampling = True
                        use_random_sample = None
                    elif use_random_sample > 100.0:
                        print(f"WARNING: use_random_sample > 100%: {use_random_sample}, capping at 100%")
                        use_random_sample = 100.0
                        use_hash_sampling = False
                    else:
                        use_hash_sampling = False
                else:  # Threshold to 100% of table - use hash-based sampling
                    use_hash_sampling = True
                    use_random_sample = None  # Don't use TABLESAMPLE when using hash sampling
            
            # setup a query manager
            qm = QueryManager(
                mat_db_name,
                segmentation_source=pcg_table_name,
                meta_db_name=aligned_volume,
                split_mode=split_mode,
                random_sample=use_random_sample,
                direct_sql_pandas=direct_sql_pandas
            )
            qm.configure_query(user_data)
            qm.apply_filter({user_data["table"]: {"valid": True}}, qm.apply_equal_filter)
            
            # Apply hash-based sampling if determined above
            if use_hash_sampling:
                qm.add_hash_spatial_sampling(
                    table_name=hash_config["table_name"],
                    spatial_column=hash_config["spatial_column"], 
                    max_points=max_points,  # Use the max_points from QUERY_LIMIT_SIZE
                    total_row_count=mat_row_count
                )
            
            # return the result
            df, column_names = qm.execute_query(
                desired_resolution=user_data["desired_resolution"]
            )
            df, warnings = update_rootids(df, user_data["timestamp"], query_map, cg_client)
            if "limit" in user_data:
                if len(df) >= user_data["limit"]:
                    warnings.append(
                        f"result has {len(df)} entries, which is equal or more \
        than limit of {user_data['limit']} there may be more results which are not shown"
                    )
            if not direct_sql_pandas:
                warnings.append(sql_query_warning)
            return df, column_names, warnings
        else:
            return None, {}, []



def execute_production_query(
    aligned_volume_name: str,
    segmentation_source: str,
    user_data: dict,
    chosen_timestamp: datetime.datetime,
    cg_client,
    allow_missing_lookups: bool = False,
    direct_sql_pandas: bool = False,
) -> pd.DataFrame:
    """_summary_

    Args:
        datastack (str): _description_
        user_data (dict): _description_
        timestamp_start (datetime.datetime): _description_
        timestamp_end (datetime.datetime): _description_

    Returns:
        pd.DataFrame: dataframe of query
        dict: _map of table name to column name mappings
        dict: list of warnings
    """
    user_timestamp = user_data["timestamp"]
    if chosen_timestamp < user_timestamp:
        query_forward = True
        start_time = chosen_timestamp
        end_time = user_timestamp
    elif chosen_timestamp > user_timestamp:
        query_forward = False
        start_time = user_timestamp
        end_time = chosen_timestamp
    else:
        abort(400, "do not use live live query to query a materialized timestamp")

    # setup a query manager on production database with split tables
    qm = QueryManager(
        aligned_volume_name,
        segmentation_source,
        split_mode=True,
        split_mode_outer=True,
        direct_sql_pandas=direct_sql_pandas
    )
    user_data_modified = strip_root_id_filters(user_data)

    qm.configure_query(user_data_modified)
    qm.select_column(user_data["table"], "created")
    qm.select_column(user_data["table"], "deleted")
    qm.select_column(user_data["table"], "superceded_id")
    qm.apply_table_crud_filter(user_data["table"], start_time, end_time)
    df, column_names = qm.execute_query(
        desired_resolution=user_data["desired_resolution"]
    )

    df, warnings = update_rootids(
        df, user_timestamp, {}, cg_client, allow_missing_lookups
    )
    if "limit" in user_data:
        if len(df) >= user_data["limit"]:
            warnings.append(
                f"result has {len(df)} entries, which is equal or more \
    than limit of {user_data['limit']} there may be more results which are not shown"
            )
    return df, column_names, warnings


def apply_filters(df, user_data, column_names):
    filter_in_dict = user_data.get("filter_in_dict", None)
    filter_out_dict = user_data.get("filter_out_dict", None)
    filter_equal_dict = user_data.get("filter_equal_dict", None)
    filter_greater_dict = user_data.get("filter_greater_dict", None)
    filter_less_dict = user_data.get("filter_less_dict", None)
    filter_greater_equal_dict = user_data.get("filter_greater_equal_dict", None)
    filter_less_equal_dict = user_data.get("filter_less_equal_dict", None)

    if filter_in_dict:
        for table, filter in filter_in_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname].isin(val)]
    if filter_out_dict:
        for table, filter in filter_in_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[~df[colname].isin(val)]
    if filter_equal_dict:
        for table, filter in filter_equal_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname] == val]
    if filter_greater_dict:
        for table, filter in filter_greater_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname] > val]
    if filter_less_dict:
        for table, filter in filter_less_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname] < val]
    if filter_greater_equal_dict:
        for table, filter in filter_greater_equal_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname] >= val]
    if filter_less_equal_dict:
        for table, filter in filter_less_equal_dict.items():
            for col, val in filter.items():
                colname = column_names[table][col]
                df = df[df[colname] <= val]
    return df


def combine_queries(
    mat_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    chosen_version: AnalysisVersion,
    user_data: dict,
    column_names: dict,
    column_names_prod: dict,
) -> pd.DataFrame:
    """combine a materialized query with an production query
       will remove deleted rows from materialized query,
       strip deleted entries from prod_df remove any CRUD columns
       and then append the two dataframes together to be a coherent
       result.

    Args:
        mat_df (pd.DataFrame): _description_
        prod_df (pd.DataFrame): _description_
        user_data (dict): _description_
        

    Returns:
        pd.DataFrame: _description_
    """
    crud_columns, created_columns = collect_crud_columns(column_names=column_names_prod)
    if mat_df is not None:
        if len(mat_df) == 0:
            if prod_df is None:
                return mat_df.drop(columns=crud_columns, axis=1, errors="ignore")
            else:
                mat_df = None
    user_timestamp = user_data["timestamp"]
    chosen_timestamp = pytz.utc.localize(chosen_version.time_stamp)
    table = user_data["table"]
    if mat_df is not None:
        mat_df = mat_df.set_index(column_names[table]["id"])
    if prod_df is not None:
        prod_df = prod_df.set_index(column_names[table]["id"])
    if (prod_df is None) and (mat_df is None):
        abort(400, f"This query on table {user_data['table']} returned no results")

    # if there is nothing to combine, just return the prod table to reflect 
    # schema with no rows
    if (mat_df is None) and len(prod_df)==0:
        cut_prod_df = prod_df.drop(crud_columns, axis=1,errors="ignore")
        if len(created_columns) > 0:
            cut_prod_df = cut_prod_df.drop(created_columns, axis=1,errors="ignore")
        return cut_prod_df.reset_index()

    if prod_df is not None:
        # if we are moving forward in time
        if chosen_timestamp < user_timestamp:
            deleted_between = (
                prod_df[column_names_prod[table]["deleted"]] > chosen_timestamp
            ) & (prod_df[column_names_prod[table]["deleted"]] < user_timestamp)
            created_between = (
                prod_df[column_names_prod[table]["created"]] > chosen_timestamp
            ) & (prod_df[column_names_prod[table]["created"]] < user_timestamp)

            to_delete_in_mat = deleted_between & ~created_between
            to_add_in_mat = created_between & ~deleted_between
            if len(prod_df[deleted_between].index) > 0:
                cut_prod_df = prod_df.drop(prod_df[deleted_between].index, axis=0)
            else:
                cut_prod_df = prod_df
        else:
            deleted_between = (
                prod_df[column_names_prod[table]["deleted"]] > user_timestamp
            ) & (prod_df[column_names_prod[table]["deleted"]] < chosen_timestamp)
            created_between = (
                prod_df[column_names_prod[table]["created"]] > user_timestamp
            ) & (prod_df[column_names_prod[table]["created"]] < chosen_timestamp)
            to_delete_in_mat = created_between & ~deleted_between
            to_add_in_mat = deleted_between & ~created_between
            if len(prod_df[created_between].index) > 0:
                cut_prod_df = prod_df.drop(prod_df[created_between].index, axis=0)
            else:
                cut_prod_df = prod_df
        # # delete those rows from materialized dataframe

        cut_prod_df = cut_prod_df.drop(crud_columns, axis=1)

        if mat_df is not None:
            created_columns = [c for c in created_columns if c not in mat_df]
            if len(created_columns) > 0:
                cut_prod_df = cut_prod_df.drop(created_columns, axis=1)

            if len(prod_df[to_delete_in_mat].index) > 0:
                mat_df = mat_df.drop(
                    prod_df[to_delete_in_mat].index, axis=0, errors="ignore"
                )
            comb_df = pd.concat([cut_prod_df, mat_df])
        else:
            comb_df = prod_df[to_add_in_mat].drop(
                columns=crud_columns, axis=1, errors="ignore"
            )
    else:
        comb_df = mat_df.drop(columns=crud_columns, axis=1, errors="ignore")

    return comb_df.reset_index()


@client_bp.expect(metadata_parser)
@client_bp.route("/datastack/<string:datastack_name>/versions")
class DatastackVersions(Resource):
    method_decorators = [
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("datastack_versions", security="apikey")
    def get(self, datastack_name: str):
        """get available versions

        Args:
            datastack_name (str): datastack name

        Returns:
            list(int): list of versions that are available
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with db_manager.session_scope(aligned_volume_name) as session:
            query = session.query(AnalysisVersion.version).filter(
                AnalysisVersion.datastack == datastack_name
            )
            args = metadata_parser.parse_args()
            if not args.get("expired"):
                query = query.filter(AnalysisVersion.valid == True)

            version_tuples = query.all()
            versions = [v[0] for v in version_tuples] 
        return versions, 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>"
)
class DatastackVersion(Resource):
    method_decorators = [
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("version metadata", security="apikey")
    def get(self, datastack_name: str, version: int):
        """get version metadata

        Args:
            datastack_name (str): datastack name
            version (int): version number

        Returns:
            dict: metadata dictionary for this version
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with db_manager.session_scope(aligned_volume_name) as session:
            analysis_version_obj = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.datastack == datastack_name)
                .filter(AnalysisVersion.version == version)
                .first()
            )
            if analysis_version_obj is None:
                return "No version found", 404
            schema = AnalysisVersionSchema()
            result = schema.dump(analysis_version_obj)
        return result, 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/table/<string:table_name>/count"
)
class FrozenTableCount(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("table count", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        table_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """get annotation count in table

        Args:
            datastack_name (str): datastack name of table
            version (int): version of table
            table_name (str): table name

        Returns:
            int: number of rows in this table
        """

        validate_table_args([table_name], target_datastack, target_version)
        db_name = f"{datastack_name}__mat{version}"

        # if the database is a split database get a split model
        # and if its not get a flat model

        with db_manager.session_scope(db_name) as session:
            mat_row_count = (
                session.query(MaterializedMetadata.row_count)
                .filter(MaterializedMetadata.table_name == table_name)
                .scalar()
            )

        return mat_row_count, 200


class CustomResource(Resource):
    @staticmethod
    def apply_decorators(*decorators):
        def wrapper(func):
            for decorator in reversed(decorators):
                func = decorator(func)
            return func

        return wrapper


@client_bp.expect(metadata_parser)
@client_bp.route("/datastack/<string:datastack_name>/metadata", strict_slashes=False)
class DatastackMetadata(Resource):
    method_decorators = [
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("all valid version metadata", security="apikey")
    def get(self, datastack_name: str):
        """get materialized metadata for all valid versions
        Args:
            datastack_name (str): datastack name
        Returns:
            list: list of metadata dictionaries
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with db_manager.session_scope(aligned_volume_name) as session:
            query = session.query(AnalysisVersion).filter(
                AnalysisVersion.datastack == datastack_name
            )
            args = metadata_parser.parse_args()
            if not args.get("expired"):
                query = query.filter(AnalysisVersion.valid == True)

            analysis_versions = query.all()

            if not analysis_versions:
                return "No valid versions found", 404
            schema = AnalysisVersionSchema()
            result = schema.dump(analysis_versions, many=True)
        return result, 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/tables"
)
class FrozenTableVersions(Resource):
    method_decorators = [
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_frozen_tables", security="apikey")
    def get(self, datastack_name: str, version: int):
        """get frozen tables

        Args:
            datastack_name (str): datastack name
            version (int): version number

        Returns:
            list(str): list of frozen tables in this version
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with db_manager.session_scope(aligned_volume_name) as session:
            av = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.version == version)
                .filter(AnalysisVersion.datastack == datastack_name)
                .first()
            )
            if av is None:
                return None, 404
            response = (
                session.query(AnalysisTable.table_name)
                .filter(AnalysisTable.analysisversion_id == av.id)
                .filter(AnalysisTable.valid == True)
                .all()
            )
          
            table_names = [r[0] for r in response]

        if not table_names:
            return None, 404 
            
        return table_names, 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/tables/metadata"
)
class FrozenTablesMetadata(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_frozen_tables_metadata", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """get frozen tables metadata

        Args:
            datastack_name (str): datastack name
            version (int): version number


        Returns:
            dict: dictionary of table metadata
        """

        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            target_datastack
        )
 
        analysis_version, analysis_tables = get_analysis_version_and_tables(
            target_datastack, target_version, aligned_volume_name
        )
        if not analysis_tables:
            
            return [], 404
        

        with request_db_session(aligned_volume_name) as db:
            for table in analysis_tables:

                table_name = table["table_name"]
                ann_md = db.database.get_table_metadata(table_name)
                # the get_table_metadata function joins on the segmentationmetadata which
                # has the segmentation_table in the table_name and the annotation table name in the annotation_table
                # field.  So when we update here, we overwrite the table_name with the segmentation table name,
                # which was not the intent of the API.
                ann_table = ann_md.pop("annotation_table", None)
                if ann_table:
                    ann_md["table_name"] = ann_table
                ann_md.pop("id")
                ann_md.pop("deleted")
                table.update(ann_md)

        return analysis_tables, 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/table/<string:table_name>/metadata"
)
class FrozenTableMetadata(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_frozen_table_metadata", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        table_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """get frozen table metadata

        Args:
            datastack_name (str): datastack name
            version (int): version number
            table_name (str): table name

        Returns:
            dict: dictionary of table metadata
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            target_datastack
        )
    
        analysis_version, analysis_table = get_analysis_version_and_table(
            target_datastack, table_name, target_version, aligned_volume_name
        )
        if not analysis_table:
            return f"No table named '{table_name}' found for version '{target_version}' in datastack '{target_datastack}'", 404

 

        with request_db_session(aligned_volume_name) as db:
            ann_md = db.database.get_table_metadata(table_name)
            if ann_md is None:
                return (
                    f"No metadata found for table named {table_name} in version {version}",
                    404,
                )
            # the get_table_metadata function joins on the segmentationmetadata which
            # has the segmentation_table in the table_name and the annotation table name in the annotation_table
            # field.  So when we update here, we overwrite the table_name with the segmentation table name,
            # which was not the intent of the API.
            ann_table = ann_md.pop("annotation_table", None)
            if ann_table:
                ann_md["table_name"] = ann_table
            ann_md.pop("id")
            ann_md.pop("deleted")
            analysis_table.update(ann_md)
        return analysis_table, 200


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/table/<string:table_name>/query"
)
class FrozenTableQuery(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("simple_query", security="apikey")
    @accepts("SimpleQuerySchema", schema=SimpleQuerySchema, api=client_bp)
    def post(
        self,
        datastack_name: str,
        version: int,
        table_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for doing a query with filters

        Args:
            datastack_name (str): datastack name
            version (int): version number
            table_name (str): table name

        Payload:
        All values are optional.  Limit has an upper bound set by the server.
        Consult the schema of the table for column names and appropriate values
        {
            "filter_out_dict": {
                "tablename":{
                    "column_name":[excluded,values]
                }
            },
            "offset": 0,
            "limit": 200000,
            "desired_resolution: [x,y,z],
            "select_columns": ["column","names"],
            "filter_in_dict": {
                "tablename":{
                    "column_name":[included,values]
                }
            },
            "filter_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_spatial_dict": {
                "tablename": {
                    "column_name": [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            },
            "filter_regex_dict": {
                "tablename": {
                    "column_name": "regex"
            }
        }
        Returns:
            pyarrow.buffer: a series of bytes that can be deserialized using pyarrow.deserialize
        """
        args = query_parser.parse_args()
        data = request.parsed_obj
        return handle_simple_query(
            datastack_name,
            version,
            table_name,
            target_datastack,
            target_version,
            args,
            data,
            convert_desired_resolution=True,
        )


def process_fields(df, fields, column_names, tags, bool_tags, numerical):
    for field_name, field in fields.items():
        col = column_names[field_name]

        if (
            field_name.endswith("_supervoxel_id")
            or field_name.endswith("_root_id")
            or field_name == "id"
            or field_name == "valid"
            or field_name == "target_id"
        ):
            continue
        if col not in df.columns:
            continue
        if isinstance(field, mm_fields.String):
            # check that this column is not all nulls
            tags.append(col)
        elif isinstance(field, mm_fields.Boolean):
            df[col] = df[col].astype(bool)
            bool_tags.append(col)
        elif isinstance(field, PostGISField):
            numerical.append(col + "_x")
            numerical.append(col + "_y")
            numerical.append(col + "_z")
        elif isinstance(field, mm_fields.Number):
            numerical.append(col)
    return df

def process_view_columns(df, model, column_names, tags, bool_tags, numerical):
    for table_column_name, table_column in model.columns.items():
        col = column_names[model.name][table_column.key]
        if (
            table_column_name.endswith("_supervoxel_id")
            or table_column_name.endswith("_root_id")
            or table_column_name == "id"
            or table_column_name == "valid"
            or table_column_name == "target_id"
        ):
            continue

        if isinstance(table_column.type, String):
            if df[col].isnull().all():
                continue
            # check that this column is not all nulls
            tags.append(col)
        elif isinstance(table_column.type, Boolean):
            if df[col].isnull().all():
                continue
            df[col] = df[col].astype(bool)
            bool_tags.append(col)
        elif isinstance(table_column.type, PostGISField):
            # if all the values are NaNs skip this column
            if df[col + "_x"].isnull().all():
                continue
            numerical.append(col + "_x")
            numerical.append(col + "_y")
            numerical.append(col + "_z")
        elif isinstance(table_column.type, (Numeric, Integer, Float)):
            if df[col].isnull().all():
                continue
            numerical.append(col)



def preprocess_dataframe(df, table_name, aligned_volume_name, column_names):
    with request_db_session(aligned_volume_name) as db:
        # check if this is a reference table
        table_metadata = db.database.get_table_metadata(table_name)
        schema = db.schema.get_flattened_schema(table_metadata["schema_type"])
        fields = schema._declared_fields

        ref_fields = None
        ref_table = None
        if table_metadata["reference_table"]:
            ref_table = table_metadata["reference_table"]
            ref_table_metadata = db.database.get_table_metadata(ref_table)
            ref_schema = db.schema.get_flattened_schema(ref_table_metadata["schema_type"])
            ref_fields = ref_schema._declared_fields

        # find the first column that ends with _root_id using next
        try:
            root_id_col = next(
                (col for col in df.columns if col.endswith("_root_id")), None
            )
        except StopIteration:
            raise ValueError("No root_id column found in dataframe")

        # pick only the first row with each root_id
        # df = df.drop_duplicates(subset=[root_id_col])
        # drop any row with root_id =0
        df = df[df[root_id_col] != 0]

        # iterate through the columns and put them into
        # categories of 'tags' for strings, 'numerical' for numbers

        tags = []
        numerical = []
        bool_tags = []

        df = process_fields(df, fields, column_names[table_name], tags, bool_tags, numerical)

        if table_metadata["reference_table"]:
            df = process_fields(
                df,
                ref_fields,
                column_names[ref_table],
                tags,
                bool_tags,
                numerical,
            )
    # Look across the tag columns and make sure that there are no
    # duplicate string values across distinct columns
    unique_vals = {}
    for tag in tags:
        unique_vals[tag] = df[tag].unique()
        unique_vals[tag] = sorted(unique_vals[tag][~pd.isnull(unique_vals[tag])])

    if len(tags) > 0:
        # find all the duplicate values across columns
        vals, counts = np.unique(
            np.concatenate([v for v in unique_vals.values()]), return_counts=True
        )
        duplicates = vals[counts > 1]

    # iterate through the tags and replace any duplicate
    # values in the dataframe with a unique value,
    # based on preprending the column name
    for tag in tags:
        for dup in duplicates:
            if dup in unique_vals[tag]:
                df[tag] = df[tag].replace(dup, f"{tag}:{dup}")

    return df, tags, bool_tags, numerical, root_id_col


def preprocess_view_dataframe(df, view_name, db_name, column_names):
    with request_db_session(db_name) as db:
        # check if this is a reference table
        view_table = db.database.get_view_table(view_name)

        # find the first column that ends with _root_id using next
        try:
            root_id_col = next(
                (col for col in df.columns if col.endswith("_root_id")), None
            )
        except StopIteration:
            raise ValueError("No root_id column found in dataframe")

        # pick only the first row with each root_id
        # df = df.drop_duplicates(subset=[root_id_col])
        # drop any row with root_id =0
        df = df[df[root_id_col] != 0]

        # iterate through the columns and put them into
        # categories of 'tags' for strings, 'numerical' for numbers

        tags = []
        numerical = []
        bool_tags = []

        process_view_columns(df, view_table, column_names, tags, bool_tags, numerical)

    # Look across the tag columns and make sure that there are no
    # duplicate string values across distinct columns
    unique_vals = {}
    for tag in tags:
        unique_vals[tag] = df[tag].unique()
        # remove nan values from unique values
        unique_vals[tag] = unique_vals[tag][~pd.isnull(unique_vals[tag])]

    if len(unique_vals) > 0:
        # find all the duplicate values across columns
        vals, counts = np.unique(
            np.concatenate([v for v in unique_vals.values()]), return_counts=True
        )
        duplicates = vals[counts > 1]

        # iterate through the tags and replace any duplicate
        # values in the dataframe with a unique value,
        # based on preprending the column name
        for tag in tags:
            for dup in duplicates:
                if dup in unique_vals[tag]:
                    df[tag] = df[tag].replace(dup, f"{tag}:{dup}")
    return df, tags, bool_tags, numerical, root_id_col


@client_bp.expect(query_seg_prop_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/table/<string:table_name>/info"
)
class MatTableSegmentInfo(Resource):
    method_decorators = [
        cached(LRUCache(maxsize=256)),
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    def get(
        self,
        datastack_name: str,
        version: int,
        table_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for getting a segment properties object for a table

        Args:
            datastack_name (str): datastack name
            version (int): version number
            table_name (str): table name

        Returns:
            json: a precomputed json object with the segment info for this table
        """
        # check how many rows are in this table
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )

        validate_table_args([table_name], target_datastack, target_version)
        db_name = f"{datastack_name}__mat{version}"

        args = query_seg_prop_parser.parse_args()
        label_format = args.get("label_format", None)
        label_columns = args.get("label_columns", None)

        # if the database is a split database get a split model
        # and if its not get a flat model
        with db_manager.session_scope(db_name) as session:

            mat_row_count = (
                session.query(MaterializedMetadata.row_count)
                .filter(MaterializedMetadata.table_name == table_name)
                .scalar()
            )
        if mat_row_count > current_app.config["QUERY_LIMIT_SIZE"]:
            return "Table too large to return info", 400
        else:
            with request_db_session(aligned_volume_name) as db:
                # check if this is a reference table
                table_metadata = db.database.get_table_metadata(table_name)

            if table_metadata["reference_table"]:
                ref_table = table_metadata["reference_table"]
                suffix_map = {table_name: "", ref_table: "_ref"}
                tables = [[table_name, "target_id"], [ref_table, "id"]]
                data = {
                    "tables": tables,
                    "suffix_map": suffix_map,
                    "desired_resolution": [1, 1, 1],
                }

                df, warnings, column_names = generate_complex_query_dataframe(
                    datastack_name,
                    version,
                    target_datastack,
                    target_version,
                    {"direct_sql_pandas": True},
                    data,
                    convert_desired_resolution=True,
                )

            else:
                df, warnings, column_names = generate_simple_query_dataframe(
                    datastack_name,
                    version,
                    table_name,
                    target_datastack,
                    target_version,
                    {"direct_sql_pandas": True},
                    {"desired_resolution": [1, 1, 1]},
                    convert_desired_resolution=True,
                )
            df = fix_dataframe_types(df)
            df, tags, bool_tags, numerical, root_id_col = preprocess_dataframe(
                df, table_name, aligned_volume_name, column_names
            )
            if label_columns is None:
                if label_format is None:
                    label_columns = "id"
            seg_prop = nglui.segmentprops.SegmentProperties.from_dataframe(
                df,
                id_col=root_id_col,
                tag_value_cols=tags,
                tag_bool_cols=bool_tags,
                number_cols=numerical,
                label_col=label_columns,
                label_format_map=label_format,
            )
            dfjson = json.dumps(seg_prop.to_dict(), cls=current_app.json_encoder)
            response = Response(dfjson, status=200, mimetype="application/json")
            return after_request(response)


@client_bp.expect(query_seg_prop_parser)
@client_bp.route("/datastack/<string:datastack_name>/table/<string:table_name>/info")
class MatTableSegmentInfoLive(Resource):
    method_decorators = [
        cached(TTLCache(maxsize=256, ttl=120)),
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    def get(
        self,
        datastack_name: str,
        table_name: str,
        version: int = 0,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for getting a segment properties object for a table
           based on a live query
        Args:
            datastack_name (str): datastack name
            table_name (str): table name

        Returns:
            json: a precomputed json object with the segment info for this table
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with request_db_session(aligned_volume_name) as db:
            # check if this is a reference table
            table_metadata = db.database.get_table_metadata(table_name)

        user_data = {
            "table": table_name,
            "timestamp": datetime.datetime.now(tz=pytz.utc),
            "suffixes": {table_name: ""},
            "desired_resolution": [1, 1, 1],
        }

        if table_metadata["reference_table"]:
            ref_table = table_metadata["reference_table"]
            user_data["suffixes"][ref_table] = "_ref"
            user_data["join_tables"] = [[table_name, "target_id", ref_table, "id"]]

        return_vals = assemble_live_query_dataframe(
            user_data, datastack_name=datastack_name, args={"direct_sql_pandas": True}
        )
        df, column_names, _, _, _ = return_vals
        df = fix_dataframe_types(df)
        vals = preprocess_dataframe(df, table_name, aligned_volume_name, column_names)
        df, tags, bool_tags, numerical, root_id_col = vals

        # parse the args
        args = query_seg_prop_parser.parse_args()
        label_format = args.get("label_format", None)
        label_columns = args.get("label_columns", None)
        if label_format is None:
            if label_columns is None:
                label_columns = "id"

        seg_prop = nglui.segmentprops.SegmentProperties.from_dataframe(
            df,
            id_col=root_id_col,
            tag_value_cols=tags,
            tag_bool_cols=bool_tags,
            number_cols=numerical,
            label_col=label_columns,
            label_format_map=label_format,
        )
        dfjson = json.dumps(seg_prop.to_dict(), cls=current_app.json_encoder)
        response = Response(dfjson, status=200, mimetype="application/json")
        return after_request(response)


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/query"
)
class FrozenQuery(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("complex_query", security="apikey")
    @accepts("ComplexQuerySchema", schema=ComplexQuerySchema, api=client_bp)
    def post(
        self,
        datastack_name: str,
        version: int,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for doing a query with filters and joins

        Args:
            datastack_name (str): datastack name
            version (int): version number

        Payload:
        All values are optional.  Limit has an upper bound set by the server.
        Consult the schema of the table for column names and appropriate values
        {
            "tables":[["table1", "table1_join_column"],
                      ["table2", "table2_join_column"]],
            "filter_out_dict": {
                "tablename":{
                    "column_name":[excluded,values]
                }
            },
            "offset": 0,
            "limit": 200000,
            "select_columns": [
                "column","names"
            ],
            "filter_in_dict": {
                "tablename":{
                    "column_name":[included,values]
                }
            },
            "filter_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_spatial_dict": {
                "tablename":{
                    "column_name":[[min_x,min_y,minz], [max_x_max_y_max_z]]
                }
            },
            "filter_regex_dict": {
                "tablename":{
                    "column_name": "regex"
                }
            }
        }
        Returns:
            pyarrow.buffer: a series of bytes that can be deserialized using pyarrow.deserialize
        """

        args = query_parser.parse_args()
        data = request.parsed_obj
        return handle_complex_query(
            datastack_name,
            version,
            target_datastack,
            target_version,
            args,
            data,
            convert_desired_resolution=True,
        )


@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/unique_string_values"
)
class TableUniqueStringValues(Resource):
    method_decorators = [
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_unique_string_values", security="apikey")
    def get(self, datastack_name: str, table_name: str):
        """get unique string values for a table

        Args:
            datastack_name (str): datastack name
            table_name (str): table name

        Returns:
            list: list of unique string values
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with request_db_session(aligned_volume_name) as db:
            unique_values = db.database.get_unique_string_values(table_name)
        return unique_values, 200


def assemble_live_query_dataframe(user_data, datastack_name, args):
    user_data["limit"] = min(
        current_app.config["QUERY_LIMIT_SIZE"],
        user_data.get("limit", current_app.config["QUERY_LIMIT_SIZE"]),
    )
    past_ver, future_ver, aligned_vol = get_closest_versions(
        datastack_name, user_data["timestamp"]
    )
    with request_db_session(aligned_vol) as db:
        check_read_permission(db, user_data["table"])
    allow_invalid_root_ids = args.get("allow_invalid_root_ids", False)
    # TODO add table owner warnings
    # if has_joins:
    #    abort(400, "we are not supporting joins yet")
    # if future_ver is None and has_joins:
    #    abort(400, 'we do not support joins when there is no future version')
    # elif has_joins:
    #     # run a future to past map version of the query
    #     check_joins(joins)
    #     chosen_version = future_ver
    if (past_ver is None) and (future_ver is None):
        abort(
            400,
            "there is no future or past version for this timestamp, is materialization broken?",
        )
    elif past_ver is not None:
        chosen_version = past_ver
    else:
        chosen_version = future_ver

    loc_cv_time_stamp = chosen_version["time_stamp"]
    loc_cv_parent_version_id = chosen_version.get("parent_version")
    loc_cv_datastack_name = chosen_version["datastack"]
    loc_cv_version_number = chosen_version["version"]
    loc_cv_is_merged = chosen_version["is_merged"]
    if isinstance(loc_cv_time_stamp, str):
        loc_cv_time_stamp = datetime.datetime.fromisoformat(loc_cv_time_stamp)

    chosen_timestamp_utc = pytz.utc.localize(loc_cv_time_stamp)
    
    effective_datastack_name_for_mat_query = datastack_name

    # map public version datastacks to their private versions
    if loc_cv_parent_version_id is not None:
        target_datastack_for_parent_lookup = loc_cv_datastack_name 
        with db_manager.session_scope(aligned_vol) as session:
            target_parent_version_obj = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.id == loc_cv_parent_version_id)
                .one_or_none()
            )
            if target_parent_version_obj:
                effective_datastack_name_for_mat_query = target_parent_version_obj.datastack
                
                newest_version_in_parent_datastack = (
                    session.query(AnalysisVersion)
                    .filter(AnalysisVersion.datastack == effective_datastack_name_for_mat_query)
                    .order_by(AnalysisVersion.time_stamp.desc())
                    .first()
                )
                if newest_version_in_parent_datastack and user_data["timestamp"] > pytz.utc.localize(newest_version_in_parent_datastack.time_stamp):
                    user_data["timestamp"] = pytz.utc.localize(newest_version_in_parent_datastack.time_stamp)
            else:
                current_app.logger.warning(f"Parent version ID {loc_cv_parent_version_id} not found in DB {aligned_vol} for datastack {target_datastack_for_parent_lookup}")


    aligned_volume_for_mat_query, pcg_table_name_for_mat_query = get_relevant_datastack_info(effective_datastack_name_for_mat_query)
    cg_client = chunkedgraph_cache.get_client(pcg_table_name_for_mat_query)

    meta_db_aligned_volume, _ = get_relevant_datastack_info(datastack_name) 
    with request_db_session(meta_db_aligned_volume) as meta_db:
        md = meta_db.database.get_table_metadata(user_data["table"])
        if not user_data.get("desired_resolution", None):
            des_res = [
                md["voxel_resolution_x"],
                md["voxel_resolution_y"],
                md["voxel_resolution_z"],
            ]
            user_data["desired_resolution"] = des_res

    modified_user_data, query_map, remap_warnings = remap_query(
        user_data,
        chosen_timestamp_utc, 
        cg_client,
        allow_invalid_root_ids,
    )
    direct_sql_pandas = args.get("direct_sql_pandas", False)
    mat_df, column_names, mat_warnings = execute_materialized_query(
        effective_datastack_name_for_mat_query, 
        aligned_volume_for_mat_query,           
        loc_cv_version_number,                  
        pcg_table_name_for_mat_query,           
        modified_user_data,
        query_map,
        cg_client,                              
        random_sample=args.get("random_sample", None),
        split_mode=not loc_cv_is_merged,
        direct_sql_pandas=direct_sql_pandas,
    )
    
    prod_df = None
    prod_warnings = []
    column_names_prod = {} 

    last_modified_in_md_utc = pytz.utc.localize(md["last_modified"])
    if (last_modified_in_md_utc > chosen_timestamp_utc) or (last_modified_in_md_utc > user_data["timestamp"]):
        prod_aligned_volume, prod_pcg_table_name = get_relevant_datastack_info(datastack_name) 
        
        cg_client_for_prod = cg_client
        if pcg_table_name_for_mat_query != prod_pcg_table_name:
            cg_client_for_prod = chunkedgraph_cache.get_client(prod_pcg_table_name)

        prod_df, column_names_prod, prod_warnings = execute_production_query(
            prod_aligned_volume,
            prod_pcg_table_name,
            user_data,
            chosen_timestamp_utc,
            cg_client_for_prod,
            args.get("allow_missing_lookups", True),
            direct_sql_pandas=direct_sql_pandas,
        )
        if mat_df is None and prod_df is not None:
            column_names = column_names_prod
    
    class MinimalChosenVersion:
        def __init__(self, naive_timestamp):
            self.time_stamp = naive_timestamp

    naive_ts_for_combine = loc_cv_time_stamp 
    minimal_chosen_version_for_combine = MinimalChosenVersion(naive_ts_for_combine)

    df = combine_queries(mat_df, prod_df, minimal_chosen_version_for_combine, user_data, column_names, column_names_prod)
    df = apply_filters(df, user_data, column_names)
    
    final_remap_warnings = remap_warnings if isinstance(remap_warnings, list) else ([remap_warnings] if remap_warnings else [])
    final_mat_warnings = mat_warnings if isinstance(mat_warnings, list) else ([mat_warnings] if mat_warnings else [])
    final_prod_warnings = prod_warnings if isinstance(prod_warnings, list) else ([prod_warnings] if prod_warnings else [])
    if not direct_sql_pandas:
        final_mat_warnings.append(sql_query_warning)
    # we want to drop columns that the user didn't ask for (mostly supervoxel columns)
    filter_column_names = copy.deepcopy(column_names)
    if user_data.get('select_columns', None) is not None:
        for table, column_map in column_names.items():
            for col, df_col_name in column_map.items():
                if col not in user_data['select_columns'][table]:
                    # then we want to drop this column
                    df.drop(df_col_name, axis=1, inplace=True)
                    filter_column_names[table].pop(col)
    return df, filter_column_names, final_mat_warnings, final_prod_warnings, final_remap_warnings


@client_bp.expect(query_parser)
@client_bp.route("/datastack/<string:datastack_name>/query")
class LiveTableQuery(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("v2_query", security="apikey")
    @accepts("V2QuerySchema", schema=V2QuerySchema, api=client_bp)
    def post(self, datastack_name: str,
        version: int = 0,
        target_version: int = None,
        target_datastack: str = None):
        """endpoint for doing a query with filters

        Args:
            datastack_name (str): datastack name

        Payload:
        All values are optional.  Limit has an upper bound set by the server.
        Consult the schema of the table for column names and appropriate values
        {
            "table":"table_name",
            "joins":[[table_name, table_column, joined_table, joined_column],
                     [joined_table, joincol2, third_table, joincol_third]]
            "timestamp": datetime.datetime.utcnow(),
            "offset": 0,
            "limit": 200000,
            "suffixes":{
                "table_name":"suffix1",
                "joined_table":"suffix2",
                "third_table":"suffix3"
            },
            "select_columns": {
                "table_name":[ "column","names"]
            },
            "filter_in_dict": {
                "table_name":{
                    "column_name":[included,values]
                }
            },
            "filter_out_dict": {
                "table_name":{
                    "column_name":[excluded,values]
                }
            },
            "filter_equal_dict": {
                "table_name":{
                    "column_name":value
                }
            },
            "filter_greater_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_spatial_dict": {
                "table_name": {
                "column_name": [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            },
            "filter_regex_dict":{
                "table_name":{
                    "column_name": "regex"
                }
            }
        }
        Returns:
            pyarrow.buffer: a series of bytes that can be deserialized using pyarrow.deserialize
        """
        args = query_parser.parse_args()
        user_data = request.parsed_obj
        return_vals = assemble_live_query_dataframe(user_data, datastack_name, args)

        df, column_names, mat_warnings, prod_warnings, remap_warnings = return_vals
        return create_query_response(
            df,
            warnings=remap_warnings + mat_warnings + prod_warnings,
            column_names=column_names,
            desired_resolution=user_data["desired_resolution"],
            return_pyarrow=args["return_pyarrow"],
            arrow_format=args["arrow_format"],
            ipc_compress=args["ipc_compress"],
        )


def find_position_prefixes(df):
    """
    Find common prefixes in a DataFrame for which there are x, y, z components.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.

    Returns:
        set: A set of prefixes that have x, y, and z components.
    """
    # Dictionary to track components for each prefix
    prefix_map = {}

    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y") or col.endswith("_z"):
            # Extract the prefix by removing the suffix
            prefix = col.rsplit("_", 1)[0]
            if not (
                prefix in ["ctr_pt_position", "bb_start_position", "bb_end_position"]
            ):
                if prefix not in prefix_map:
                    prefix_map[prefix] = set()
                # Add the component (x, y, or z) to the prefix
                prefix_map[prefix].add(col.split("_")[-1])

    # Find prefixes that have all three components (x, y, z)
    position_prefixes = {
        prefix
        for prefix, components in prefix_map.items()
        if {"x", "y", "z"}.issubset(components)
    }
    columns = []
    # get the set of columns that are covered by the prefixes
    for prefix in position_prefixes:
        for suffix in ["_x", "_y", "_z"]:
            col = f"{prefix}{suffix}"
            if col in df.columns:
                columns.append(col)
    return position_prefixes, columns


@cached(LRUCache(maxsize=256))
def get_precomputed_properties_and_relationships(datastack_name, table_name):
    """Get precomputed relationships from the database.

    Args:
        db: The database connection.
        table_name (str): The name of the table.
    """

    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    
    past_version, _, _ = get_closest_versions(
        datastack_name, datetime.datetime.now(tz=pytz.utc)
    )
    mat_db_name = f"{datastack_name}__mat{past_version['version']}"
    with db_manager.session_scope(mat_db_name) as session:    
        mat_row_count = (
            session.query(MaterializedMetadata.row_count)
            .filter(MaterializedMetadata.table_name == table_name)
            .scalar()
        )
    with request_db_session(aligned_volume_name) as db:
        # check if this is a reference table
        table_metadata = db.database.get_table_metadata(table_name)
    # convert timestamp string to timestamp object
    # with UTC timezone
    if isinstance(past_version['time_stamp'], str):
        past_version['time_stamp'] = datetime.datetime.fromisoformat(
            past_version['time_stamp']
        )
    timestamp = past_version['time_stamp']
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)

    user_data = {
        "table": table_name,
        "timestamp": timestamp,
        "suffixes": {table_name: ""},
        "desired_resolution": [1, 1, 1],
        "limit": 1,
    }

    if table_metadata["reference_table"]:
        ref_table = table_metadata["reference_table"]
        user_data["suffixes"][ref_table] = "_ref"
        user_data["join_tables"] = [[table_name, "target_id", ref_table, "id"]]

    return_vals = assemble_live_query_dataframe(
        user_data, datastack_name=datastack_name, args={"direct_sql_pandas": True}
    )
    df, column_names, mat_warnings, prod_warnings, remap_warnings = return_vals
    df = fix_dataframe_types(df)

    relationships = []
    for c in df.columns:
        if c.endswith("pt_root_id"):
            relationships.append(c)
    unique_values = db.database.get_unique_string_values(table_name)

    properties = []
    # find all the columns that start with the same thing
    # but end with _pt_position_{x,y,z}
    geometry_prefixes, geometry_columns = find_position_prefixes(df)

    for c in df.columns:
        if c.endswith("id"):
            continue
        elif c.endswith("valid"):
            continue
        elif c.endswith("pt_root_id"):
            continue
        if c in geometry_columns:
            continue
        elif c in unique_values.keys():
            prop = viewer_state.AnnotationPropertySpec(
                id=c,
                type="uint32",
                enum_values=np.arange(0, len(unique_values[c])),
                enum_labels=sorted(unique_values[c]),
            )
        else:
            if df[c].dtype == "float64":
                type = "float32"
            elif df[c].dtype == "float32":
                type = "float32"
            elif df[c].dtype == "bool":
                type = "int32"
            elif df[c].dtype == "int64":
                type = "int32"
            elif df[c].dtype == "int32":
                type = "int32"
            else:
                continue
            prop = viewer_state.AnnotationPropertySpec(id=c, type=type, description=c)
        properties.append(prop)

    if len(geometry_prefixes) == 0:
        abort(400, "No geometry columns found for table {}".format(table_name))
    if len(geometry_prefixes) == 1:
        ann_type = "point"
    elif len(geometry_prefixes) == 2:
        ann_type = "line"
    else:
        abort(400, "More than 2 geometry columns found for table {}".format(table_name))

    return relationships, properties, list(geometry_prefixes), column_names, ann_type, mat_row_count


bounds_cache = LRUCache(maxsize=128)


@cached(bounds_cache)
def get_precomputed_bounds(datastack_name):
    """Get precomputed bounds from the database.

    Args:
        datastack_name: The datastack name.
        table_name (str): The name of the table.

    Returns:
        dict: the bounds for the precomputed table.
    """
    ds_info = get_datastack_info(datastack_name)
    img_source = ds_info["segmentation_source"]
    cv = cloudvolume.CloudVolume(img_source, use_https=True)
    bbox = cv.bounds * cv.resolution
    lower_bound = bbox.minpt.tolist()
    upper_bound = bbox.maxpt.tolist()
    return lower_bound, upper_bound

def _cache_key_spatial_levels(total_size, annotation_count, target_limit=10000):
    """Create a hashable cache key for spatial index level calculation."""
    return (tuple(total_size.tolist()), annotation_count, target_limit)

@cached(LRUCache(maxsize=128), key=_cache_key_spatial_levels)
def calculate_spatial_index_levels(total_size, annotation_count, target_limit=10000):
    """
    Calculate the number of spatial index levels needed based on uniform distribution assumption.
    
    Uses isotropic chunking following Neuroglancer spec: each successive level applies
    all subdivisions that improve isotropy compared to the original state. This can
    result in multiple dimensions being subdivided simultaneously per level.
    
    Always adds a final 'spatial_high_res' level with approximately 15000x15000x2000 
    chunk sizes for high-resolution queries.
    
    Args:
        total_size: numpy array of [width, height, depth] of the bounding box
        annotation_count: total number of annotations
        target_limit: maximum annotations per grid cell at finest level
    
    Returns:
        list: List of spatial index level configurations, including final spatial_high_res level
    """
    if annotation_count <= target_limit:
        # If we have few enough annotations, use overview + high-res levels
        levels = [{
            "key": "spatial_overview",
            "grid_shape": [1, 1, 1],
            "chunk_size": total_size.tolist(),
            "limit": target_limit
        }]
        
        return levels
    
    levels = []
    current_grid_shape = np.array([1, 1, 1], dtype=int)
    level = 0
    
    while True:
        # Calculate chunk size for current grid shape
        current_chunk_size = total_size / current_grid_shape
        
        # Calculate total number of grid cells at this level
        total_cells = np.prod(current_grid_shape)
        
        # Estimate annotations per cell (assuming uniform distribution)
        annotations_per_cell = annotation_count / total_cells
        
        # Add this level
        level_key = "spatial_overview" if level == 0 else f"spatial_level_{level}"
        levels.append({
            "key": level_key,
            "grid_shape": current_grid_shape.tolist(),
            "chunk_size": current_chunk_size.astype(int).tolist(),
            "limit": target_limit
        })
        
        # Check if we're fine enough - if average annotations per cell is acceptable
        if annotations_per_cell <= target_limit:
            break
            
        # For more isotropic chunking, subdivide the largest dimensions
        # that don't make isotropy significantly worse
        next_grid_shape = current_grid_shape.copy()
        
        def calculate_isotropy_metric(chunk_size):
            """Calculate isotropy metric - lower is more isotropic."""
            return np.max(chunk_size) / np.min(chunk_size)
        
        original_isotropy = calculate_isotropy_metric(current_chunk_size)
        
        made_change = False
        # Test subdividing the dimensions with the largest chunk sizes
        # but only if it doesn't make isotropy much worse
        dim_order = np.argsort(current_chunk_size)[::-1]  # Largest first
        
        for dim in dim_order:
            # Test doubling the grid in this dimension (halving chunk size)
            test_grid_shape = next_grid_shape.copy()
            test_grid_shape[dim] *= 2
            test_chunk_size = total_size / test_grid_shape
            test_isotropy = calculate_isotropy_metric(test_chunk_size)
            
            # Subdivide if it improves isotropy or doesn't make it much worse
            # Also prioritize subdividing large dimensions to avoid very elongated chunks
            max_chunk = np.max(current_chunk_size)
            
            # Determine if we should subdivide this dimension
            should_subdivide = False
            
            # Case 1: Dimension is large and isotropy doesn't get too bad
            if (current_chunk_size[dim] >= max_chunk * 0.8 and  # Dimension is large
                test_isotropy <= original_isotropy * 1.5):      # Isotropy doesn't get too bad
                should_subdivide = True
            
            # Case 2: Fallback for isotropic volumes - if annotations/cell still too high
            # and we have good isotropy, subdivide any dimension that doesn't make it much worse
            elif (annotations_per_cell > target_limit * 1.1 and  # Still over target
                  original_isotropy < 2.0 and                   # Already fairly isotropic
                  test_isotropy <= original_isotropy * 1.2):     # Don't make isotropy much worse
                should_subdivide = True
            
            if should_subdivide:
                next_grid_shape[dim] *= 2
                made_change = True
        
        # If no beneficial subdivision found, stop
        if not made_change:
            break
            
        current_grid_shape = next_grid_shape
        level += 1
        
        # Safety check to prevent infinite loops
        if level > 10:
            break
    
    # # Add a final high-resolution level with target chunk size of approximately [15000, 15000, 2000]
    # # This level divides the volume into chunks that are suitable for high-resolution queries
    # target_chunk_size = np.array([15000, 15000, 2000], dtype=float)
    
    # # Calculate grid shape needed to achieve target chunk sizes (rounded up)
    # high_res_grid_shape = np.maximum([1, 1, 1], np.ceil(total_size / target_chunk_size).astype(int))
    
    # # Calculate actual chunk size that divides evenly into the volume
    # high_res_chunk_size = total_size / high_res_grid_shape
    
    # # Calculate total number of grid cells and annotations per cell
    # high_res_total_cells = np.prod(high_res_grid_shape)
    # high_res_annotations_per_cell = annotation_count / high_res_total_cells
    
    # # Add the high-resolution level
    # levels.append({
    #     "key": "spatial_high_res",
    #     "grid_shape": high_res_grid_shape.tolist(),
    #     "chunk_size": high_res_chunk_size.astype(int).tolist(),
    #     "limit": target_limit
    # })
    
    return levels



def get_precomputed_info(datastack_name, table_name):
    """Get precomputed properties from the database.

    Args:
        datastack_name: The datastack name.
        table_name (str): The name of the table.

    Returns:
        dict: the info file for the precomputed table.
        
    Note:
        Uses dynamic spatial index level calculation based on annotation distribution
        and configurable target limits for optimal Neuroglancer performance.
    """
   

    vals = get_precomputed_properties_and_relationships(datastack_name, table_name)
    relationships, properties, geometry_columns, column_names, ann_type, mat_row_count = vals

    lower_bound, upper_bound = get_precomputed_bounds(datastack_name)
    total_size = np.array(upper_bound) - np.array(lower_bound)
    
    # Use dynamic spatial index level calculation
    target_limit = current_app.config.get("PRECOMPUTED_SPATIAL_INDEX_LIMIT", 10000)
    spatial_keys = calculate_spatial_index_levels(
        total_size=total_size,
        annotation_count=mat_row_count,
        target_limit=target_limit
    )

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {
            "x": [np.float64(1e-09), "m"],
            "y": [np.float64(1e-09), "m"],
            "z": [np.float64(1e-09), "m"],
        },
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "annotation_type": ann_type,
        "properties": [p.to_json() for p in properties],
        "relationships": [{"id": r, "key": f"{r}"} for r in relationships],
        "by_id": {"key": "by_id"},
        "spatial": spatial_keys
    }
    return metadata


@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/info"
)
class LiveTablePrecomputedInfo(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_precomputed_info", security="apikey")
    def get(
        self,
        datastack_name: str,
        table_name: str,
        version: int = 0,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """get precomputed info for a table

        Args:
            datastack_name (str): datastack name
            table_name (str): table name
            target_datastack (str): target datastack name
            target_version (int): target version number
            version (int): version number
        Returns:
            dict: dictionary of precomputed info
        """
        # validate_table_args([table_name], target_datastack, target_version)
        info = get_precomputed_info(datastack_name, table_name)

        return info, 200

def query_spatial_no_filter(
    datastack_name: str,
    table_name: str,
    lower_bound: np.array,
    upper_bound: np.array,
    timestamp: datetime.datetime = None,
    volume_fraction: float = 1.0,
    sampling: bool = True,
):
    """get precomputed annotation by id

    Args:
        datastack_name (str): datastack name
        table_name (str): table name
        lower_bound (np.array, optional): lower bound of the bounding box.
        upper_bound (np.array, optional): upper bound of the bounding box.
         Defaults to None in which case will use the bounds of the datastack.
         units should be in nanometers
        timestamp (datetime.datetime, optional): timestamp to use for the query.
         Defaults to None in which case will use the latest timestamp of root_id
        volume_fraction (float, optional): fraction of the volume this represents.
        sampling (bool, optional): whether to apply spatial sampling.

    Returns:
        pd.DataFrame: dataframe of precomputed properties with grid-based spatial sampling
         applied to limit results to QUERY_LIMIT_SIZE for better performance
    """
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    
    with request_db_session(aligned_volume_name) as db:
        table_metadata = db.database.get_table_metadata(table_name)
        

        vals = get_precomputed_properties_and_relationships(datastack_name, table_name)
        relationships, properties, d, column_names, ann_type, mat_row_count = vals

        timestamp = datetime.datetime.now(tz=pytz.utc) if timestamp is None else timestamp
        
        spatial_column = None
        spatial_table = None
        vox_res = None
        
        # Find the best spatial column using improved selection logic
        def select_best_spatial_column(fields, table_name, metadata):
            """Select the best spatial column, prioritizing those with corresponding root_id fields."""
            from materializationengine.utils import make_root_id_column_name
            
            spatial_columns = []
            all_field_names = set(fields.keys())
            
            # Collect all PostGIS fields
            for field_name, field in fields.items():
                if isinstance(field, PostGISField):
                    # Check if this spatial column has a corresponding root_id field
                    try:
                        root_id_name = field_name[:-9] + "_root_id"
                        has_root_id = root_id_name in all_field_names
                    except (IndexError, AttributeError):
                        # If root_id name construction fails, assume no root_id
                        has_root_id = False
                    
                    spatial_columns.append({
                        'field_name': field_name,
                        'table_name': table_name,
                        'metadata': metadata,
                        'has_root_id': has_root_id
                    })
            
            # Sort by priority: root_id fields first, then by order found
            spatial_columns.sort(key=lambda x: (not x['has_root_id'], x['field_name']))
            
            return spatial_columns[0] if spatial_columns else None
        
        table_schema = db.schema.get_flattened_schema(table_metadata["schema_type"])
        table_fields = table_schema._declared_fields
        
        # Try to find spatial column in main table
        best_spatial = select_best_spatial_column(table_fields, table_name, table_metadata)
        if best_spatial:
            spatial_column = best_spatial['field_name']
            spatial_table = best_spatial['table_name']
            vox_res = np.array([
                table_metadata["voxel_resolution_x"],
                table_metadata["voxel_resolution_y"],
                table_metadata["voxel_resolution_z"],
            ])
            
        user_data = {
            "table": table_name,
            "timestamp": timestamp,
            "suffixes": {table_name: ""},
            "desired_resolution": [1, 1, 1],
        }
        
        if table_metadata["reference_table"]:
            ref_table = table_metadata["reference_table"]
            user_data["suffixes"][ref_table] = "_ref"
            user_data["join_tables"] = [[table_name, "target_id", ref_table, "id"]]
            # find the spatial column in the reference table
            if spatial_column is None:
                # get the reference table schema
                ref_metadata = db.database.get_table_metadata(ref_table)
                ref_schema = db.schema.get_flattened_schema(ref_metadata["schema_type"])
                ref_fields = ref_schema._declared_fields
                
                # Use the same improved selection logic for reference table
                best_ref_spatial = select_best_spatial_column(ref_fields, ref_table, ref_metadata)
                if best_ref_spatial:
                    spatial_column = best_ref_spatial['field_name']
                    spatial_table = best_ref_spatial['table_name']
                    vox_res = np.array([
                        ref_metadata["voxel_resolution_x"],
                        ref_metadata["voxel_resolution_y"],
                        ref_metadata["voxel_resolution_z"],
                    ])
                
    if (spatial_column is None):
        abort(400, f"No spatial column found for table {table_name}")
    
    if lower_bound is not None:
        lower_bound = (np.array(lower_bound) / vox_res).astype(int)
        upper_bound = (np.array(upper_bound) / vox_res).astype(int)
        user_data["filter_spatial_dict"] = {
            spatial_table: {
                spatial_column: [lower_bound.tolist(), upper_bound.tolist()]
            }
        }
    
    # Add hash sampling configuration if we have spatial info
    if spatial_column and spatial_table and sampling:
        # Get the target limit for spatial sampling
        max_points = current_app.config.get("QUERY_LIMIT_SIZE", 10000)
        
        user_data["hash_sampling_config"] = {
            "enabled": True,
            "table_name": spatial_table,
            "spatial_column": spatial_column,
            "volume_fraction": volume_fraction,
            "max_points": max_points,
        }
    

    return_vals = assemble_live_query_dataframe(
        user_data, datastack_name=datastack_name, args={"direct_sql_pandas": True}
    )
    df, column_names, mat_warnings, prod_warnings, remap_warnings = return_vals
    df = fix_dataframe_types(df)
    return df

def query_by_id(
    datastack_name: str,
    table_name: str,
    annotation_id: int,
    timestamp: datetime.datetime = None,
):
    """get precomputed annotation by id

    Args:
        datastack_name (str): datastack name
        table_name (str): table name
        annotation_id (int): annotation id
        timestamp (datetime.datetime, optional): timestamp to use for the query.
         Defaults to None in which case will use the latest timestamp of root_id

    Returns:
        pd.DataFrame: dataframe of precomputed properties
    """
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    with request_db_session(aligned_volume_name) as db:
        table_metadata = db.database.get_table_metadata(table_name)

    vals = get_precomputed_properties_and_relationships(datastack_name, table_name)
    relationships, properties, geometry_columns, column_names, ann_type, mat_row_count = vals

    timestamp = datetime.datetime.now(tz=pytz.utc) if timestamp is None else timestamp
    user_data = {
        "table": table_name,
        "timestamp": timestamp,
        "suffixes": {table_name: ""},
        "filter_equal_dict": {
            table_name: {"id": annotation_id},
        },
        "desired_resolution": [1, 1, 1],
    }
    
    if table_metadata["reference_table"]:
        ref_table = table_metadata["reference_table"]
        user_data["suffixes"][ref_table] = "_ref"
        user_data["join_tables"] = [[table_name, "target_id", ref_table, "id"]]

    return_vals = assemble_live_query_dataframe(
        user_data, datastack_name=datastack_name, args={"direct_sql_pandas": True}
    )
    df, column_names, mat_warnings, prod_warnings, remap_warnings = return_vals
    df = fix_dataframe_types(df)
    return df

def live_query_by_relationship(
    datastack_name: str,
    table_name: str,
    column_name: str,
    segid: int,
    timestamp: datetime.datetime = None,
):
    """get precomputed relationships for a table

    Args:
        datastack_name (str): datastack name
        table_name (str): table name
        column_name (str): column name
        segid (int): segment id
        timestamp (datetime.datetime, optional): timestamp to use for the query.
         Defaults to None in which case will use the latest timestamp of root_id


    Returns:
        pd.DataFrame: dataframe of precomputed relationships
    """
    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)
    with request_db_session(aligned_volume_name) as db:
        table_metadata = db.database.get_table_metadata(table_name)
 
    vals = get_precomputed_properties_and_relationships(datastack_name, table_name)
    relationships, properties, geometry_columns, column_names, ann_type, mat_row_count = vals

    filter_table = None
    for table in column_names:
        for col in column_names[table]:
            if col == column_name:
                filter_table = table
                break
    if filter_table is None:
        abort(400, "column_name not found in table {}".format(table_name))
    if timestamp is None:
        cg_client = chunkedgraph_cache.get_client(pcg_table_name)
        timestamp = cg_client.get_root_timestamps(segid, latest=True)[0]
    user_data = {
        "table": table_name,
        "timestamp": timestamp,
        "suffixes": {table_name: ""},
        "filter_equal_dict": {
            filter_table: {column_name: segid},
        },
        "desired_resolution": [1, 1, 1],
    }
    if table_metadata["reference_table"]:
        ref_table = table_metadata["reference_table"]
        user_data["suffixes"][ref_table] = "_ref"
        user_data["join_tables"] = [[table_name, "target_id", ref_table, "id"]]

    return_vals = assemble_live_query_dataframe(
        user_data, datastack_name=datastack_name, args={"direct_sql_pandas": True}
    )
    df, column_names, mat_warnings, prod_warnings, remap_warnings = return_vals
    df = fix_dataframe_types(df)
    return df


def format_df_to_bytes(df, datastack_name, table_name, encode_single=False):
    """format the dataframe to bytes

    Args:
        df (pd.DataFrame): dataframe
        datastack_name (str): datastack name
        table_name (str): table name

    Returns:
        bytes: byte stream of dataframe
    """
    vals = get_precomputed_properties_and_relationships(datastack_name, table_name)
    relationships, properties, geometry_columns, column_names, anntype, mat_row_count = vals
    writer = AnnotationWriter(
        anntype, relationships=relationships, properties=properties
    )
    if anntype == "point":
        point_cols = [
            geometry_columns[0] + "_x",
            geometry_columns[0] + "_y",
            geometry_columns[0] + "_z",
        ]
    elif anntype == "line":
        pointa_cols = [
            geometry_columns[0] + "_x",
            geometry_columns[0] + "_y",
            geometry_columns[0] + "_z",
        ]
        pointb_cols = [
            geometry_columns[1] + "_x",
            geometry_columns[1] + "_y",
            geometry_columns[1] + "_z",
        ]
    else:
        abort("400", "Unsupported annotation type {}".format(anntype))
    for p in properties:
        if p.enum_values is not None:
            df[p.id].replace(
                {label: val for label, val in zip(p.enum_labels, p.enum_values)},
                inplace=True,
            )
    for i, row in df.iterrows():
        kwargs = {p.id: df.loc[i, p.id] for p in properties}
        if encode_single:
            # update kwargs with relationships
            for r in relationships:
                if r in df.columns:
                    kwargs[r] = df.loc[i, r]
        if row.get("valid", True):
            if anntype == "point":
                point = df.loc[i, point_cols].tolist()
                writer.add_point(point, id=df.loc[i, "id"], **kwargs)
            elif anntype == "line":
                point_a = df.loc[i, pointa_cols].tolist()
                point_b = df.loc[i, pointb_cols].tolist()
                writer.add_line(point_a, point_b, id=df.loc[i, "id"], **kwargs)
    if encode_single:
        output_bytes = writer._encode_single_annotation(writer.annotations[0])
    else:
        output_bytes = writer._encode_multiple_annotations(writer.annotations)

    return output_bytes

@client_bp.route(
    "/datastack/<string:datastack_name>/table/"
)
class LiveTablesAvailable(Resource):
    method_decorators = [
        validate_datastack,
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_live_tables", security="apikey")
    def get(self, datastack_name: str, version: int =-1, target_datastack: str = None, target_version: int =None, **args):
        """get live tables for a datastack

        Args:
            datastack_name (str): datastack name

        Returns:
            HTML directory listing of available tables
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        with db_manager.session_scope(aligned_volume_name) as session:
            version = session.query(AnalysisVersion).filter(
                    AnalysisVersion.datastack == target_datastack
                ).order_by(
                    AnalysisVersion.time_stamp.desc()
                ).first()
            if version is not None:
                tables = session.query(AnalysisTable).filter(
                    AnalysisTable.analysisversion_id == version.id,
                    AnalysisTable.valid == True
                ).all()
                tables = [table.table_name for table in tables]
        
        # Generate HTML directory listing
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Index of /datastack/{datastack_name}/table/</title>
    <style>
        body {{ font-family: monospace; margin: 40px; }}
        h1 {{ font-size: 18px; margin-bottom: 20px; }}
        a {{ text-decoration: none; color: #0066cc; }}
        a:hover {{ text-decoration: underline; }}
        .file {{ display: block; padding: 2px 0; }}
        .parent {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Index of /datastack/{datastack_name}/table/</h1>
    <a href="../" class="file parent">[Parent Directory]</a>
"""
        
        # Add each table as a directory entry
        for table in sorted(tables):
            html_content += f'    <a href="{table}/precomputed/" class="file">{table}/</a>\n'
        
        html_content += """</body>
</html>"""
        
        return Response(html_content, mimetype='text/html')



@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/"
)
class LiveTablesAvailable(Resource):
    method_decorators = [
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_info_listing", security="apikey")
    def get(self, datastack_name: str, table_name: str):
        """get info listing

        Args:
            datastack_name (str): datastack name
            table_name (str): table name

        Returns:
            HTML directory pointing to info file
        """
        
        # Generate HTML directory listing
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Index of /datastack/{datastack_name}/table/</title>
    <style>
        body {{ font-family: monospace; margin: 40px; }}
        h1 {{ font-size: 18px; margin-bottom: 20px; }}
        a {{ text-decoration: none; color: #0066cc; }}
        a:hover {{ text-decoration: underline; }}
        .file {{ display: block; padding: 2px 0; }}
        .parent {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Index of /datastack/{datastack_name}/table/{table_name}/precomputed/</h1>
    <a href="../" class="file parent">[Parent Directory]</a>
"""
        
        # Add each table as a directory entry
        html_content += f'    <a href="info" class="file">info/</a>\n'
        html_content += f'    <a href="by_id" class="file">by_id/</a>\n'
        html_content += f'    <a href="spatial" class="file">spatial/</a>\n'
        html_content += """</body>
</html>"""
        
        return Response(html_content, mimetype='text/html')


# General spatial endpoint that handles all spatial levels (spatial_overview, spatial_level_1, spatial_level_2, etc.)
# This replaces the old spatial_high_res endpoint and provides a unified interface for all spatial levels

# Cache for spatial query results with 20-minute TTL
_spatial_bytes_cache = TTLCache(maxsize=1000, ttl=1200)  # 20 minutes = 1200 seconds

def _cache_key_spatial_bytes(datastack_name, table_name, spatial_level, x_bin, y_bin, z_bin, timestamp):
    """Generate cache key for spatial bytes result."""
    # Round timestamp to minute precision to improve cache hit rate
    timestamp_str = None
    if timestamp is not None:
        timestamp_rounded = timestamp.replace(second=0, microsecond=0)
        timestamp_str = timestamp_rounded.isoformat()
    
    return hashkey(datastack_name, table_name, spatial_level, x_bin, y_bin, z_bin, timestamp_str)

@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/<string:spatial_level>/<int:x_bin>_<int:y_bin>_<int:z_bin>"
)
class LiveTableSpatialLevel(Resource):
    method_decorators = [
        validate_datastack,
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_precomputed spatial level cutout", security="apikey")
    def get(self, datastack_name: str, table_name: str, spatial_level: str, x_bin: int, y_bin: int, z_bin: int, version: int = 0, target_datastack: str = None, target_version: int = None):
        """get precomputed spatial cutout for a table at any spatial level

        This is a general endpoint that works with all spatial levels defined in the 
        precomputed info. It dynamically determines the chunk size and grid bounds
        based on the spatial level configuration.

        Args:
            datastack_name (str): datastack name
            table_name (str): table name
            spatial_level (str): spatial level key (e.g., 'spatial_overview', 'spatial_level_1', 'spatial_level_2', etc.)
                                Must match a key from the spatial index configuration
            x_bin (int): x bin index for spatial grid coordinate (0-based)
            y_bin (int): y bin index for spatial grid coordinate (0-based)  
            z_bin (int): z bin index for spatial grid coordinate (0-based)
            version (int): version number (ignored)
            target_datastack (str): target datastack name (ignored)
            target_version (int): target version number (ignored)

        Query Parameters:
            None - Intelligent spatial sampling is automatically applied based on the 
            volume fraction and grid density. Fine-grained chunks use less sampling,
            while coarse chunks use more aggressive sampling for optimal performance.

        Returns:
            bytes: byte stream of precomputed spatial data with adaptive sampling
            
        Example URLs:
            .../precomputed/spatial_overview/0_0_0
            .../precomputed/spatial_level_1/0_0_0  
            .../precomputed/spatial_level_1/1_2_0
            .../precomputed/spatial_level_2/4_3_1
        """
        precomputed_info = get_precomputed_info(datastack_name, table_name)

        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            target_datastack
        )
        if version is not None:
            with db_manager.session_scope(aligned_volume_name) as session:
                analysis_version = session.query(AnalysisVersion).filter(
                    AnalysisVersion.datastack == datastack_name,
                    AnalysisVersion.version == version,
                ).one_or_none()
                timestamp = analysis_version.time_stamp.astimezone(datetime.timezone.utc) if analysis_version else None
        else:
            timestamp = None
        
        # get the info for this spatial index from the precomputed info
        spatial_keys = precomputed_info.get("spatial", [])
        spatial_key = None
        for spatial_index in spatial_keys:
            if spatial_index["key"] == spatial_level:
                spatial_key = spatial_index
                break
        
        if spatial_key is None:
            abort(404, f"Spatial level '{spatial_level}' not found for table '{table_name}'")
        
        # Validate grid coordinates are within valid bounds
        grid_shape = spatial_key.get("grid_shape", [1, 1, 1])
        if (x_bin < 0 or x_bin >= grid_shape[0] or
            y_bin < 0 or y_bin >= grid_shape[1] or 
            z_bin < 0 or z_bin >= grid_shape[2]):
            abort(400, f"Grid coordinates ({x_bin}, {y_bin}, {z_bin}) are out of bounds for spatial level '{spatial_level}' with grid shape {grid_shape}")
        
        lower_bound, upper_bound = get_precomputed_bounds(datastack_name)

        chunk_size = np.array(spatial_key["chunk_size"])
        grid_shape = np.array(spatial_key["grid_shape"])
        
        # Calculate what fraction of the total volume this chunk represents
        total_volume = np.prod(np.array(upper_bound) - np.array(lower_bound))
        chunk_volume = np.prod(chunk_size)
        
        # Validate volume calculations to prevent invalid tablesample parameters
        if total_volume <= 0:
            print(f"WARNING: Invalid total_volume: {total_volume}, bounds: {lower_bound} to {upper_bound}")
            volume_fraction = 1.0  # Fallback to no sampling
        elif chunk_volume <= 0:
            print(f"WARNING: Invalid chunk_volume: {chunk_volume}, chunk_size: {chunk_size}")
            volume_fraction = 1.0  # Fallback to no sampling
        else:
            volume_fraction = chunk_volume / total_volume
            
        # Ensure volume_fraction is valid for tablesample
        if not np.isfinite(volume_fraction) or volume_fraction <= 0:
            print(f"WARNING: Invalid volume_fraction: {volume_fraction}, defaulting to 1.0")
            volume_fraction = 1.0
        elif volume_fraction > 1.0:
            print(f"WARNING: volume_fraction > 1.0: {volume_fraction}, capping at 1.0")
            volume_fraction = 1.0
        
        # get the lower and upper bounds of this grid
        lower_bound = np.array(lower_bound) + np.array(
            [x_bin * chunk_size[0], y_bin * chunk_size[1], z_bin * chunk_size[2]]
        )
        upper_bound = lower_bound + chunk_size
        if "high_res" in spatial_level:
            sampling=False
        else:
            sampling = True
        
        # Check cache first for this specific spatial chunk
        cache_key = _cache_key_spatial_bytes(datastack_name, table_name, spatial_level, x_bin, y_bin, z_bin, timestamp)
        
        if cache_key in _spatial_bytes_cache:
            bytes_data = _spatial_bytes_cache[cache_key]
        else:
            # Query and format data if not in cache
            df = query_spatial_no_filter(datastack_name,
                                         table_name,
                                         lower_bound,
                                         upper_bound,
                                         timestamp,
                                         volume_fraction=volume_fraction,
                                         sampling=sampling)
            
            bytes_data = format_df_to_bytes(df, datastack_name, table_name)
            
            # Cache the result
            _spatial_bytes_cache[cache_key] = bytes_data

        response= Response(bytes_data, mimetype='application/octet-stream')
        response.headers["Content-Disposition"] = (
            f"attachment; filename={datastack_name}_{table_name}_{spatial_level}_{x_bin}_{y_bin}_{z_bin}.bin"
        )
        headers = {
            "access-control-allow-credentials": "true",
            "access-control-expose-headers": "Cache-Control, Content-Disposition, Content-Encoding, Content-Length, Content-Type, Date, ETag, Server, Vary, X-Content-Type-Options, X-Frame-Options, X-Powered-By, X-XSS-Protection",
            "content-disposition": "attachment",
            "Content-Type": "application/octet-stream",
            "Content-Name": f"{datastack_name}_{table_name}_{spatial_level}_{x_bin}_{y_bin}_{z_bin}.bin",
        }
        response.headers.update(headers)
        return response


# # Backward compatibility endpoint for spatial_overview at coordinates 0_0_0
# # New code should use the general spatial endpoint: .../precomputed/spatial_overview/0_0_0
# @client_bp.route(
#     "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/spatial_overview/0_0_0"
# )
# class LiveTableSpatialOverview(Resource):
#     method_decorators = [
#         validate_datastack,
#         auth_requires_permission("view", table_arg="datastack_name"),
#         reset_auth,
#     ]

#     @client_bp.doc("get_precomputed_overview", security="apikey")
#     def get(self, datastack_name: str, table_name: str,  version: int = 0, target_datastack: str = None, target_version: int = None):
#         """get precomputed spatial overview for a table

#         Args:
#             datastack_name (str): datastack name
#             table_name (str): table name
#             version (int): version number (ignored)
#             target_datastack (str): target datastack name (ignored)
#             target_version (int): target version number (ignored)

#         Query Parameters:
#             None - grid-based spatial sampling is automatically applied using 
#             QUERY_LIMIT_SIZE from Flask config to ensure good performance

#         Returns:
#             bytes: byte stream of precomputed spatial overview with representative sampling
#         """
  
#         aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
#             target_datastack
#         )
#         if version is not None:
#             with db_manager.session_scope(aligned_volume_name) as session:
#                 analysis_version = session.query(AnalysisVersion).filter(
#                     AnalysisVersion.datastack == datastack_name,
#                     AnalysisVersion.version == version,
#                 ).one_or_none()
#                 timestamp = analysis_version.time_stamp.astimezone(datetime.timezone.utc) if analysis_version else None
#         else:
#             timestamp = None
        
#         lower_bound, upper_bound = get_precomputed_bounds(datastack_name)
        
#         df = query_spatial_no_filter(datastack_name, table_name, None, None, timestamp)

#         bytes = format_df_to_bytes(df, datastack_name, table_name)

#         response= Response(bytes, mimetype='application/octet-stream')
#         response.headers["Content-Disposition"] = (
#             f"attachment; filename={datastack_name}_{table_name}_spatial_overview.bin"
#         )
#         headers = {
#             "access-control-allow-credentials": "true",
#             "access-control-expose-headers": "Cache-Control, Content-Disposition, Content-Encoding, Content-Length, Content-Type, Date, ETag, Server, Vary, X-Content-Type-Options, X-Frame-Options, X-Powered-By, X-XSS-Protection",
#             "content-disposition": "attachment",
#             "Content-Type": "application/octet-stream",
#             "Content-Name": f"{datastack_name}_{table_name}_spatial_overview.bin",
#         }
#         response.headers.update(headers)
#         return response

@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/by_id/<int:id>"
)
class LiveTablePrecomputedById(Resource):
    method_decorators = [
        validate_datastack,
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_precomputed_by_id", security="apikey")
    def get(self, datastack_name: str, table_name: str, id: int, version: int = 0, target_datastack: str = None, target_version: int = None):
        """get precomputed by_id for a table

        Args:
            datastack_name (str): datastack name
            table_name (str): table name
            id (int): annotation id
            version (int): version number (ignored)
            target_datastack (str): target datastack name (ignored)
            target_version (int): target version number (ignored)

        Returns:
            bytes: byte stream of precomputed by_id
        """
  
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            target_datastack
        )
        if version is not None:
            with db_manager.session_scope(aligned_volume_name) as session:
                analysis_version = session.query(AnalysisVersion).filter(
                    AnalysisVersion.datastack == datastack_name,
                    AnalysisVersion.version == version,
                ).one_or_none()
                timestamp = analysis_version.time_stamp.astimezone(datetime.timezone.utc) if analysis_version else None
        else:
            timestamp = None
        
        df = query_by_id(datastack_name, table_name, id, timestamp)

        bytes = format_df_to_bytes(df, datastack_name, table_name, encode_single=True)

        response= Response(bytes, mimetype='application/octet-stream')
        response.headers["Content-Disposition"] = (
            f"attachment; filename={datastack_name}_{table_name}_{id}.bin"
        )
        headers = {
            "access-control-allow-credentials": "true",
            "access-control-expose-headers": "Cache-Control, Content-Disposition, Content-Encoding, Content-Length, Content-Type, Date, ETag, Server, Vary, X-Content-Type-Options, X-Frame-Options, X-Powered-By, X-XSS-Protection",
            "content-disposition": "attachment",
            "Content-Type": "application/octet-stream",
            "Content-Name": f"{datastack_name}_{table_name}_{id}.bin",
        }
        response.headers.update(headers)
        return response

@client_bp.route(
    "/datastack/<string:datastack_name>/table/<string:table_name>/precomputed/<string:column_name>/<int:segid>"
)
class LiveTablePrecomputedRelationship(Resource):
    method_decorators = [
        validate_datastack,
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("get_precomputed_relationships", security="apikey")
    def get(self, datastack_name: str, table_name: str, column_name: str, segid: int, version: int = 0, target_datastack: str = None, target_version: int = None):
        """get precomputed relationships for a table

        Args:
            datastack_name (str): datastack name
            table_name (str): table name
            column_name (str): column name
            segid (int): segment id
            version (int): version number (ignored)
            target_datastack (str): target datastack name (ignored)
            target_version (int): target version number (ignored)

        Returns:
            bytes: byte stream of precomputed relationships
        """
        if not column_name.endswith("pt_root_id"):
            abort(400, "column_name must end with pt_root_id")
        
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            target_datastack
        )
        if version is not None:
            with db_manager.session_scope(aligned_volume_name) as session:
                analysis_version = session.query(AnalysisVersion).filter(
                    AnalysisVersion.datastack == datastack_name,
                    AnalysisVersion.version == version,
                ).one_or_none()
                timestamp = analysis_version.time_stamp.astimezone(datetime.timezone.utc) if analysis_version else None
        else:
            timestamp = None
        df = live_query_by_relationship(datastack_name, table_name, column_name, segid, timestamp)
        bytes = format_df_to_bytes(df, datastack_name, table_name)

        # format a flask response with bytes as raw bytes conent
        response = Response(bytes, status=200, mimetype="application/octet-stream")
        response.headers["Content-Disposition"] = (
            f"attachment; filename={datastack_name}_{table_name}_{column_name}_{segid}.bin"
        )
        headers = {
            "access-control-allow-credentials": "true",
            "access-control-expose-headers": "Cache-Control, Content-Disposition, Content-Encoding, Content-Length, Content-Type, Date, ETag, Server, Vary, X-Content-Type-Options, X-Frame-Options, X-Powered-By, X-XSS-Protection",
            "content-disposition": "attachment",
            "Content-Type": "application/octet-stream",
            "Content-Name": f"{datastack_name}_{table_name}_{column_name}_{segid}.bin",
        }
        response.headers.update(headers)

        return response


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/views"
)
class AvailableViews(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("available_views", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        target_datastack: str = None,
        target_version: int = None,
    ) -> List[AnalysisViewSchema]:
        """endpoint for getting available views

        Args:
            datastack_name (str): datastack name

        Returns:
            dict: a dictionary of views
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        if version == 0:
            mat_db_name = f"{aligned_volume_name}"
        else:
            mat_db_name = f"{datastack_name}__mat{version}"
        with request_db_session(mat_db_name) as meta_db:
            views = meta_db.database.get_views(datastack_name)
            views = AnalysisViewSchema().dump(views, many=True)
        view_d = {}
        for view in views:
            name = view.pop("table_name")
            view_d[name] = view
        return view_d


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/views/<string:view_name>/metadata"
)
class ViewMetadata(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("view_metadata", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        view_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ) -> AnalysisViewSchema:
        """endpoint for getting metadata about a view

        Args:
            datastack_name (str): datastack name
            view_name (str): table names

        Returns:
            dict: a dictionary of metadata about the view
        """

        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        if version == 0:
            mat_db_name = f"{aligned_volume_name}"
        else:
            mat_db_name = f"{datastack_name}__mat{version}"
        
        md = get_cached_view_metadata(aligned_volume_name, datastack_name, view_name)

        return md


def assemble_view_dataframe(datastack_name, version, view_name, data, args):
    """
    Assemble a dataframe from a view
    Args:
        datastack_name (str): datastack name
        version (int): version number
        view_name (str): view name
        data (dict): query data
        args (dict): query arguments
    Returns:
        pd.DataFrame: dataframe
        list: column names
        list: warnings
    """

    aligned_volume_name, pcg_table_name = get_relevant_datastack_info(datastack_name)

    if version == 0:
        mat_db_name = f"{aligned_volume_name}"
    else:
        mat_db_name = f"{datastack_name}__mat{version}"

    # check_read_permission(db, table_name)

    max_limit = current_app.config.get("QUERY_LIMIT_SIZE", 200000)

    limit = data.get("limit", max_limit)
    if limit > max_limit:
        limit = max_limit

    get_count = args.get("count", False)
    if get_count:
        limit = None
    
    md = get_cached_view_metadata(aligned_volume_name, datastack_name, view_name)

    if not data.get("desired_resolution", None):
        des_res = [
            md["voxel_resolution_x"],
            md["voxel_resolution_y"],
            md["voxel_resolution_z"],
        ]
        data["desired_resolution"] = des_res
    direct_sql_pandas = args.get("direct_sql_pandas", False)
    qm = QueryManager(
        mat_db_name,
        segmentation_source=pcg_table_name,
        split_mode=False,
        limit=limit,
        offset=data.get("offset", 0),
        get_count=get_count,
        direct_sql_pandas=direct_sql_pandas
    )
    qm.add_view(datastack_name, view_name)
    qm.apply_filter(data.get("filter_in_dict", None), qm.apply_isin_filter)
    qm.apply_filter(data.get("filter_out_dict", None), qm.apply_notequal_filter)
    qm.apply_filter(data.get("filter_equal_dict", None), qm.apply_equal_filter)
    qm.apply_filter(data.get("filter_greater_dict", None), qm.apply_greater_filter)
    qm.apply_filter(data.get("filter_less_dict", None), qm.apply_less_filter)
    qm.apply_filter(
        data.get("filter_greater_equal_dict", None), qm.apply_greater_equal_filter
    )
    qm.apply_filter(
        data.get("filter_less_equal_dict", None), qm.apply_less_equal_filter
    )
    qm.apply_filter(data.get("filter_spatial_dict", None), qm.apply_spatial_filter)
    qm.apply_filter(data.get("filter_regex_dict", None), qm.apply_regex_filter)
    select_columns = data.get("select_columns", None)
    if select_columns:
        for column in select_columns:
            qm.select_column(view_name, column)
    else:
        qm.select_all_columns(view_name)

    df, column_names = qm.execute_query(desired_resolution=data["desired_resolution"])
    df.drop(columns=["deleted", "superceded"], inplace=True, errors="ignore")
    warnings = []
    if not direct_sql_pandas:
        warnings.append(sql_query_warning)
    current_app.logger.info("query: {}".format(data))
    current_app.logger.info("args: {}".format(args))
    user_id = str(g.auth_user["id"])
    current_app.logger.info(f"user_id: {user_id}")

    if len(df) == limit:
        warnings.append(f'201 - "Limited query to {limit} rows')
    warnings = update_notice_text_warnings(md, warnings, view_name)

    return df, column_names, warnings


# Define your cache (LRU Cache with a maximum size of 100 items)
view_mat_cache = LRUCache(maxsize=100)
view_live_cache = TTLCache(maxsize=100, ttl=60)
latest_mat_cache = TTLCache(maxsize=100, ttl=60 * 60 * 12)


def conditional_view_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate a cache key
        key = request.url
        if kwargs.get("version") == -1:
            # Check if the result is in the live cache
            if key in view_live_cache:
                return view_live_cache[key]
            else:
                result = func(*args, **kwargs)
                view_live_cache[key] = result
                return result

        # Check if the 'version' argument is in the kwargs and if it is set to 0
        if kwargs.get("version") == 0:
            # Check if the result is in the live cache
            if key in view_live_cache:
                return view_live_cache[key]
            else:
                result = func(*args, **kwargs)
                view_live_cache[key] = result
                return result
        else:
            # Check if the result is in the materialized cache
            if key in view_mat_cache:
                return view_mat_cache[key]
            else:
                result = func(*args, **kwargs)
                view_mat_cache[key] = result
                return result

    return wrapper


@client_bp.expect(query_seg_prop_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/view/<string:view_name>/info"
)
class MatViewSegmentInfo(Resource):
    method_decorators = [
        conditional_view_cache,
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    def get(
        self,
        datastack_name: str,
        version: int,
        view_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for getting a segment properties object for a view

        Args:
            datastack_name (str): datastack name
            version (int): version number
            view_name (str): view name

        Returns:
            json: a precomputed json object with the segment info for this view
        """
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )

        if version == -1:
            version = get_latest_version(datastack_name)
        mat_db_name = f"{datastack_name}__mat{version}"
        if version == 0:
            mat_db_name = f"{aligned_volume_name}"

        df, column_names, warnings = assemble_view_dataframe(
            datastack_name, version, view_name, {}, {"direct_sql_pandas": True}
        )

        df, tags, bool_tags, numerical, root_id_col = preprocess_view_dataframe(
            df, view_name, mat_db_name, column_names
        )

        args = query_seg_prop_parser.parse_args()
        label_format = args.get("label_format", None)
        label_columns = args.get("label_columns", None)
        if label_format is None:
            if label_columns is None:
                label_columns = df.columns[0]
        df = fix_dataframe_types(df)
        seg_prop = nglui.segmentprops.SegmentProperties.from_dataframe(
            df,
            id_col=root_id_col,
            tag_value_cols=tags,
            tag_bool_cols=bool_tags,
            number_cols=numerical,
            label_col=label_columns,
            label_format_map=label_format,
        )
        # use the current_app encoder to encode the seg_prop.to_dict()
        # to ensure that the json is serialized correctly
        dfjson = json.dumps(seg_prop.to_dict(), cls=current_app.json_encoder)
        response = Response(dfjson, status=200, mimetype="application/json")
        return after_request(response)


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/views/<string:view_name>/query"
)
class ViewQuery(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("view_query", security="apikey")
    @validate_datastack
    @accepts("SimpleQuerySchema", schema=SimpleQuerySchema, api=client_bp)
    def post(
        self,
        datastack_name: str,
        version: int,
        view_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for doing a query with filters

        Args:
            datastack_name (str): datastack name
            version (int): version number
            view_name (str): table names


        Payload:
        All values are optional.  Limit has an upper bound set by the server.
        Consult the schema of the table for column names and appropriate values
        {
            "filter_out_dict": {
                "tablename":{
                    "column_name":[excluded,values]
                }
            },
            "offset": 0,
            "limit": 200000,
            "desired_resolution: [x,y,z],
            "select_columns": ["column","names"],
            "filter_in_dict": {
                "tablename":{
                    "column_name":[included,values]
                }
            },
            "filter_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_greater_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_less_equal_dict": {
                "tablename":{
                    "column_name":value
                }
            },
            "filter_spatial_dict": {
                "tablename":{
                    "column_name":[[min_x,min_y,min_z], [max_x,max_y,max_z]]
                }
            },
            "filter_regex_dict": {
                "tablename":{
                    "column_name": "regex"
                }
            }
        }
        Returns:
            pyarrow.buffer: a series of bytes that can be deserialized using pyarrow.deserialize
        """
        args = query_parser.parse_args()
        data = request.parsed_obj
        df, column_names, warnings = assemble_view_dataframe(
            datastack_name, version, view_name, data, args
        )
        return create_query_response(
            df,
            warnings=warnings,
            column_names=column_names,
            desired_resolution=data["desired_resolution"],
            return_pyarrow=args["return_pyarrow"],
            arrow_format=args["arrow_format"],
            ipc_compress=args["ipc_compress"],
        )


def get_table_schema(table):
    """
    Get the schema of a table as a jsonschema
    Args:
        table (Table): sqlalchemy table object
    Returns:
        dict: jsonschema of the table

    """
    properties = {}

    for column in table.columns:
        column_type = None
        format = None
        if isinstance(column.type, String):
            column_type = "string"
        elif isinstance(column.type, Integer):
            column_type = "integer"
        elif isinstance(column.type, Float):
            column_type = "float"
        elif isinstance(column.type, DateTime):
            column_type = "string"
            format = "date-time"
        elif isinstance(column.type, Boolean):
            column_type = "boolean"
        elif isinstance(column.type, Geometry):
            column_type = "SpatialPoint"
        elif isinstance(column.type, Numeric):
            column_type = "number"
        else:
            raise ValueError(f"Unsupported column type: {column.type}")

        properties[column.name] = {"type": column_type}
        if format:
            properties[column.name]["format"] = format

    return properties


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/views/<string:view_name>/schema"
)
class ViewSchema(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("view_schema", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        view_name: str,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for getting schema about a view

        Args:
            datastack_name (str): datastack name
            version (int): version number
            view_name (str): table names

        Returns:
            dict: a jsonschema of the view
        """

        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )

        if version == 0:
            mat_db_name = f"{aligned_volume_name}"
        else:
            mat_db_name = f"{datastack_name}__mat{version}"
        with request_db_session(mat_db_name) as meta_db:
            table = meta_db.database.get_view_table(view_name)

        return get_table_schema(table)


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int(signed=True):version>/views/schemas"
)
class ViewSchemas(Resource):
    method_decorators = [
        validate_datastack,
        limit_by_category("fast_query"),
        auth_requires_permission("view", table_arg="datastack_name"),
        reset_auth,
    ]

    @client_bp.doc("view_schemas", security="apikey")
    def get(
        self,
        datastack_name: str,
        version: int,
        target_datastack: str = None,
        target_version: int = None,
    ):
        """endpoint for getting view schemas

        Args:
            datastack_name (str): datastack name
            version (int): version number

        Returns:
            dict: a dict of jsonschemas of each view
        """

        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )

        if version == 0:
            mat_db_name = f"{aligned_volume_name}"
        else:
            mat_db_name = f"{datastack_name}__mat{version}"
        with request_db_session(mat_db_name) as meta_db:
            views = meta_db.database.get_views(datastack_name)
        if not views:
            return {}, 404
        
        dumped_views = AnalysisViewSchema(many=True).dump(views)
        
        schemas = {}
        for view_dict in dumped_views:
            view_name = view_dict["table_name"]
            table = meta_db.database.get_view_table(view_name)
            schemas[view_name] = get_table_schema(table)
        return schemas, 200
