import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components.experimental.evaluation import \
    ModelEvaluationOp as evaluation_op
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.batch_predict_job import \
    ModelBatchPredictOp as batch_prediction_op
from google_cloud_pipeline_components.v1.model import \
    ModelUploadOp as model_upload_op
from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
                                                              ModelDeployOp)
from kfp.v2 import compiler
from kfp.v2.components import importer_node
from kfp.v2.dsl import Input, Metrics, component

PROJECT_ID = "sandbox-dev-dbg"
REGION = "europe-west1"
BUCKET_URI = "gs://test-ad789" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/bikes_weather"

UUID="test2"

DATA_URIS = [
    "gs://cloud-samples-data/vertex-ai/dataset-management/datasets/safe_driver/dataset_safe_driver_train_10k.csv"
]
MODEL_URI = "gs://cloud-samples-data/vertex-ai/google-cloud-aiplatform-ci-artifacts/models/safe_driver/model"
# Create working dir
WORKING_DIR = "{PIPELINE_ROOT}"
MODEL_DISPLAY_NAME = "safe-driver"
BATCH_PREDICTION_DISPLAY_NAME = "batch-prediction-on-pipelines-model"


@kfp.dsl.pipeline(name="upload-evaluate-")
def pipeline(
    metric: str,
    threshold: float,
    project: str = PROJECT_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    batch_prediction_display_name: str = BATCH_PREDICTION_DISPLAY_NAME,
    batch_prediction_data_uris: list = DATA_URIS,
):
    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=MODEL_URI,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "artifactUri": MODEL_URI,
            "predictSchemata": {
                "predictionSchemaUri": MODEL_URI + "/prediction_schema.yaml",
                "instanceSchemaUri": MODEL_URI + "/instance.yaml",
            },
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:prod",
                "healthRoute": "/health",
                "predictRoute": "/predict",
            },
        },
    )

    model_task = model_upload_op(
        project=project,
        display_name=model_display_name,
        unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
    )
    endpoint_create_op = EndpointCreateOp(
        project=project,
        display_name="endpoint",
    )

    ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_task.outputs["model"],
        deployed_model_display_name="test",
        dedicated_resources_machine_type="n1-standard-16",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="evaluation_demo_pipeline.json",
)

DISPLAY_NAME = "safe_driver" + UUID

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="evaluation_demo_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"metric": "auPrc", "threshold": 0.95},
    enable_caching=True,
)

job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")
