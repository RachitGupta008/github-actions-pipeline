from typing import NamedTuple

# import kfp
from google.cloud import aiplatform
from kfp import dsl
from kfp.v2.dsl import (Artifact, ClassificationMetrics, Input, Metrics,
                        Output, component)

BUCKET_URI = "gs://ad-2345" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/automlbatch"

@kfp.dsl.pipeline(name="automl-batch", pipeline_root=PIPELINE_ROOT)
def pipeline(
    
    job_display_name: str = "test",
    gcs_destination_output_uri_prefix: str = "gs://ad-2345/batchoutput",
    # model_resource_name: str, # e.g.projects/62283298672/locations/europe-west2/models/8862952125269803008
    batch_predict_gcs_source_uris: list = ["gs://ad-2345/gender_classification_test.csv"],
    
    DATASET_DISPLAY_NAME: str = "test",
    TRAINING_DISPLAY_NAME: str = "test",
    MODEL_DISPLAY_NAME: str = "test",
    project: str = "sandbox-dev-dbg",
    gcp_region: str = "europe-west1",
    
    batch_predict_predictions_format: str = "jsonl",
    batch_predict_instances_format: str = "csv",
    batch_predict_machine_type: str = "n1-standard-4",
    MACHINE_TYPE: str = "n1-standard-4",
):

    from google_cloud_pipeline_components.aiplatform import (
        AutoMLTabularTrainingJobRunOp, ModelBatchPredictOp,
        TabularDatasetCreateOp)
    from google_cloud_pipeline_components.experimental.evaluation import GetVertexModelOp

    dataset_create_op = TabularDatasetCreateOp(
        project=project, location=gcp_region, display_name=DATASET_DISPLAY_NAME, gcs_source='gs://ad-2345/automl/gender_classification.csv')

    training_op = AutoMLTabularTrainingJobRunOp(
        project=project,
        location=gcp_region,
        display_name=TRAINING_DISPLAY_NAME,
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
        budget_milli_node_hours=1000,
        model_display_name=MODEL_DISPLAY_NAME,
        column_specs={
            "Height": "auto",
            "Weight": "auto",
        },
        
        dataset=dataset_create_op.outputs["dataset"],
        target_column="Gender",
    )
    
#     get_model_task = GetVertexModelOp(model_resource_name=MODEL_DISPLAY_NAME)
    
#     get_model_task.after(training_op)
    
    batch_explain_task = ModelBatchPredictOp(
        project=project,
        location=gcp_region,
        model=training_op.outputs["model"],
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
    package_path="tabular_classification_pipeline_2.json",
)

job = aiplatform.PipelineJob(
    project="sandbox-dev-dbg",
    location="europe-west1",
    display_name="test",
    template_path="tabular_classification_pipeline_2.json",
    pipeline_root=PIPELINE_ROOT,
    
    
    enable_caching=False,
)

job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")
