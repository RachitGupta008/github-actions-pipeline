from typing import NamedTuple

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import component
BUCKET_URI = "gs://ad-2345" 
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/hw"

@dsl.component
def hello_world(text: str) -> str:
    print(text)
    return text


# compiler.Compiler().compile(hello_world, "hw.yaml")

@dsl.component
def two_outputs(
    text: str,
) -> NamedTuple(
    "Outputs",
    [
        ("output_one", str),  # Return parameters
        ("output_two", str),
    ],
):
    

    o1 = f"output one from text: {text}"
    o2 = f"output two from text: {text}"
    print("output one: {}; output_two: {}".format(o1, o2))
    return (o1, o2)

@dsl.component
def consumer(text1: str, text2: str, text3: str) -> str:
    print(f"text1: {text1}; text2: {text2}; text3: {text3}")
    return f"text1: {text1}; text2: {text2}; text3: {text3}"

@dsl.pipeline(
    name="intro-pipeline-unique",
    description="A simple intro pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def hello_pipeline(text: str = "hi there"):
    hw_task = hello_world(text=text)
    two_outputs_task = two_outputs(text=text)
    consumer_task = consumer(  # noqa: F841
        text1=hw_task.output,
        text2=two_outputs_task.outputs["output_one"],
        text3=two_outputs_task.outputs["output_two"],
    )


compiler.Compiler().compile(pipeline_func=hello_pipeline, package_path="intro_pipeline.yaml")
DISPLAY_NAME = "intro_pipeline_job_unique"

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="intro_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
)

job.run(service_account="vertex-sa@sandbox-dev-dbg.iam.gserviceaccount.com")
