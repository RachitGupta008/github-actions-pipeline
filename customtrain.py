from typing import Any, Dict, List

import google.cloud.aiplatform as aip
import kfp
from kfp import compiler

PROJECT_ID = "sandbox-dev-dbg"
REGION = "europe-west1"
BUCKET_URI = "gs://test-ad789" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/bikes_weather"
print(kfp.__version__)
MODEL_DISPLAY_NAME = "train"
@kfp.dsl.pipeline(name="train" )
def pipeline(
    project: str = PROJECT_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    
):
    
    from google_cloud_pipeline_components.v1.custom_job import \
        CustomTrainingJobOp
    

    custom_job_task = CustomTrainingJobOp(
        project=project,
        display_name="model-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    
                    "imageUri": "gcr.io/sandbox-dev-dbg/test-ad:customtrain",
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


compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="custom_train_pipeline.json",
)

DISPLAY_NAME = "custom_train"

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="custom_train_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False,
)

job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")

    
