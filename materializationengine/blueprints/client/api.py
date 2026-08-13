from dynamicannotationdb.models import AnalysisTable, AnalysisVersion
from flask import abort, request
from flask_accepts import accepts
from flask_restx import Namespace, Resource, inputs, reqparse
from middle_auth_client import auth_requires_permission

from materializationengine.blueprints.client.common import (
    get_analysis_version_and_table,
    get_flat_model,
    handle_complex_query,
    handle_simple_query,
    validate_table_args,
)
from materializationengine.blueprints.client.common import (
    unhandled_exception as common_unhandled_exception,
)
from materializationengine.blueprints.client.datastack import validate_datastack
from materializationengine.blueprints.client.schemas import (
    ComplexQuerySchema,
    SimpleQuerySchema,
)
from materializationengine.blueprints.client.utils import (
    add_warnings_to_headers,
    update_notice_text_warnings,
)
from materializationengine.blueprints.reset_auth import reset_auth
from materializationengine.database import db_manager, dynamic_annotation_cache
from materializationengine.info_client import (
    get_aligned_volumes,
    get_relevant_datastack_info,
)
from materializationengine.models import MaterializedMetadata
from materializationengine.schemas import AnalysisTableSchema, AnalysisVersionSchema

__version__ = "5.24.0"


authorizations = {
    "apikey": {"type": "apiKey", "in": "query", "name": "middle_auth_token"}
}

client_bp = Namespace(
    "Materialization Client",
    authorizations=authorizations,
    description="Materialization Client",
)


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


query_parser = reqparse.RequestParser()
query_parser.add_argument(
    "return_pyarrow",
    type=inputs.boolean,
    default=True,
    required=False,
    location="args",
    help="whether to return query in pyarrow compatible binary format (faster), false returns json",
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
    "split_positions",
    type=inputs.boolean,
    default=False,
    required=False,
    location="args",
    help="whether to return position columns as seperate x,y,z columns (faster)",
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
    "random_sample",
    type=inputs.positive,
    default=None,
    required=False,
    location="args",
    help="How many samples to randomly get using tablesample on annotation tables, useful for visualization of large tables does not work as a random sample of query",
)


def check_aligned_volume(aligned_volume):
    aligned_volumes = get_aligned_volumes()
    if aligned_volume not in aligned_volumes:
        abort(400, f"aligned volume: {aligned_volume} not valid")


@client_bp.route("/datastack/<string:datastack_name>/versions")
class DatastackVersions(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
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
            response = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.datastack == datastack_name)
                .filter(AnalysisVersion.valid == True)
                .all()
            )

            if response is None:
                return "No valid versions found", 404
            versions = [av.version for av in response]
            return versions, 200


@client_bp.route("/datastack/<string:datastack_name>/version/<int:version>")
class DatastackVersion(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
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

            response = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.datastack == datastack_name)
                .filter(AnalysisVersion.version == version)
                .first()
            )
            if response is None:
                return "No version found", 404
            schema = AnalysisVersionSchema()
            return schema.dump(response), 200


@client_bp.route("/datastack/<string:datastack_name>/metadata")
class DatastackMetadata(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
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
            response = (
                session.query(AnalysisVersion)
                .filter(AnalysisVersion.datastack == datastack_name)
                .filter(AnalysisVersion.valid == True)
                .all()
            )
            if not response:
                return "No valid versions found", 404
            
            schema = AnalysisVersionSchema()
            result = schema.dump(response, many=True)
        return result, 200


@client_bp.route("/datastack/<string:datastack_name>/version/<int:version>/tables")
class FrozenTableVersions(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
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
                abort(404, f"version {version} does not exist for {datastack_name} ")
            response = (
                session.query(AnalysisTable)
                .filter(AnalysisTable.analysisversion_id == av.id)
                .filter(AnalysisTable.valid == True)
                .all()
            )

            if response is None:
                return None, 404
            return [r.table_name for r in response], 200


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int:version>/table/<string:table_name>/metadata"
)
class FrozenTableMetadata(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
    @client_bp.doc("get_frozen_table_metadata", security="apikey")
    def get(self, datastack_name: str, version: int, table_name: str):
        """get frozen table metadata

        Args:
            datastack_name (str): datastack name
            version (int): version number
            table_name (str): table name

        Returns:
            dict: dictionary of table metadata
        """
        validate_table_args([table_name], datastack_name, version)
        aligned_volume_name, _ = get_relevant_datastack_info(
            datastack_name
        )
        
        analysis_version_dict, analysis_table_dict = get_analysis_version_and_table(
            datastack_name, table_name, version, aligned_volume_name
        )

        if analysis_version_dict is None:
            abort(404, f"Version {version} not found for datastack {datastack_name}")
        
        if analysis_table_dict is None:
            abort(404, f"Table '{table_name}' not found for version {version} in datastack {datastack_name}")

        tables_dict = analysis_table_dict

        db = dynamic_annotation_cache.get_db(aligned_volume_name)
        ann_md = db.database.get_table_metadata(table_name)
        ann_md.pop("id")
        ann_md.pop("deleted")

        warnings = update_notice_text_warnings(ann_md, [], table_name)
        headers = add_warnings_to_headers({}, warnings)

        tables_dict.update(ann_md)
        return tables_dict, 200, headers


@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int:version>/table/<string:table_name>/count"
)
class FrozenTableCount(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
    @validate_datastack
    @client_bp.doc("simple_query", security="apikey")
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
        aligned_volume_name, pcg_table_name = get_relevant_datastack_info(
            datastack_name
        )
        validate_table_args([table_name], target_datastack, target_version)
        mat_db_name = f"{datastack_name}__mat{version}"

        with db_manager.session_scope(mat_db_name) as session:
            mat_row_count = (
                session.query(MaterializedMetadata.row_count)
                .filter(MaterializedMetadata.table_name == table_name)
                .scalar()
            )

        return mat_row_count, 200


@client_bp.expect(query_parser)
@client_bp.route(
    "/datastack/<string:datastack_name>/version/<int:version>/table/<string:table_name>/query"
)
class FrozenTableQuery(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
    @validate_datastack
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
                "tablename": {
                "column_name": [[min_x, min_y, min_z], [max_x, max_y, max_z]]
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
            False,
        )


@client_bp.expect(query_parser)
@client_bp.route("/datastack/<string:datastack_name>/version/<int:version>/query")
class FrozenQuery(Resource):
    @reset_auth
    @auth_requires_permission("view", table_arg="datastack_name")
    @validate_datastack
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
            }
        }
        Returns:
            pyarrow.buffer: a series of bytes that can be deserialized using pyarrow.deserialize
        """

        args = query_parser.parse_args()
        data = request.parsed_obj
        return handle_complex_query(
            datastack_name, version, target_datastack, target_version, args, data, False
        )
