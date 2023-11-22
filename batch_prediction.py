from typing import NamedTuple

import kfp
from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2.dsl import (
                        Output, component)

BUCKET_URI = "gs://ad-2345" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/batch"

@kfp.dsl.pipeline(name="test")
def pipeline(
    project: str,
    location: str,
    job_display_name: str,
    gcs_destination_output_uri_prefix: str,
    model_resource_name: str, # e.g. projects/62283298672/locations/europe-west2/models/8862952125269803008
    batch_predict_gcs_source_uris: list, # e.g. ['gs://gcf-sources-62283298672-europe-west2/automl/gender_classification_test.csv']
    
    batch_predict_predictions_format: str = "jsonl",
    batch_predict_instances_format: str = "csv",
    batch_predict_machine_type: str = "n1-standard-4",
):

    from google_cloud_pipeline_components.aiplatform import ModelBatchPredictOp
    from google_cloud_pipeline_components.experimental.evaluation import GetVertexModelOp

    # Get the Vertex AI model resource
    get_model_task = GetVertexModelOp(model_resource_name=model_resource_name)

    # Run Batch Explanations
    batch_explain_task = ModelBatchPredictOp(
        project=project,
        location=location,
        model=get_model_task.outputs["model"],
        job_display_name=job_display_name,
        gcs_source_uris=batch_predict_gcs_source_uris,
        instances_format=batch_predict_instances_format,
        predictions_format=batch_predict_predictions_format,
        gcs_destination_output_uri_prefix=gcs_destination_output_uri_prefix,
        machine_type=batch_predict_machine_type
    )

from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="batch_prediction.json",
)

DISPLAY_NAME = "safe_driver" 

job = aiplatform.PipelineJob(
    project="sandbox-dev-dbg",
    location="europe-west1",
    display_name="test",
    template_path="batch_prediction.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
    "project":"sandbox-dev-dbg",
    "location":"europe-west1",
    "job_display_name":"test",
    "gcs_destination_output_uri_prefix":"gs://ad-2345",
    "model_resource_name": "projects/472475384454/locations/europe-west1/models/6398937771330764800",
    "batch_predict_gcs_source_uris":["gs://ad-2345/gender_classification_test.csv"]},
    
    enable_caching=False,
)

job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")
