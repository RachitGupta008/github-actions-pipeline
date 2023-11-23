from typing import Any, Dict, List

import google.cloud.aiplatform as aip
import kfp
from kfp.v2 import compiler

PROJECT_ID = "sandbox-dev-dbg"
REGION = "europe-west1"
BUCKET_URI = "gs://ad-2345" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/bikes_weather"

hp_dict: str = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
data_dir: str = "gs://aju-dev-demos-codelabs/bikes_weather/"
TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]

# create working dir to pass to job spec
WORKING_DIR = f"{PIPELINE_ROOT}"

MODEL_DISPLAY_NAME = "train_deploy"
print(TRAINER_ARGS, WORKING_DIR, MODEL_DISPLAY_NAME)


@kfp.dsl.pipeline(name="train-endpoint-deploy" )
def pipeline(
    project: str = PROJECT_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
):
    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.custom_job import \
        CustomTrainingJobOp
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
                                                              ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from kfp.v2.components import importer_node

    custom_job_task = CustomTrainingJobOp(
        project=project,
        display_name="model-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    "args": TRAINER_ARGS,
                    "env": [{"name": "AIP_MODEL_DIR", "value": WORKING_DIR}],
                    "imageUri": "gcr.io/google-samples/bw-cc-train:latest",
                },
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": "n1-standard-16",
                    "accelerator_type": aip.gapic.AcceleratorType.NVIDIA_TESLA_K80,
                    "accelerator_count": 2,
                },
            }
        ],
    )

    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=WORKING_DIR,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
            },
        },
    ).after(custom_job_task)

    # model_upload_op = ModelUploadOp(
    #     project=project,
    #     display_name=model_display_name,
    #     unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
    # )
    # model_upload_op.after(import_unmanaged_model_task)

    # endpoint_create_op = EndpointCreateOp(
    #     project=project,
    #     display_name="pipelines-created-endpoint",
    # )

    # ModelDeployOp(
    #     endpoint=endpoint_create_op.outputs["endpoint"],
    #     model=model_upload_op.outputs["model"],
    #     deployed_model_display_name=model_display_name,
    #     dedicated_resources_machine_type="n1-standard-16",
    #     dedicated_resources_min_replica_count=1,
    #     dedicated_resources_max_replica_count=1,
    # )

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="tabular_regression_pipeline.json",
)

DISPLAY_NAME = "bikes_weather_"

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="tabular_regression_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)
job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")
