from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component, 
                        OutputPath, 
                        InputPath)

from kfp.v2 import compiler
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
# from google_cloud_pipeline_components import aiplatform as gcc_aip

# PATH=%env PATH
# %env PATH={PATH}:/home/jupyter/.local/bin
REGION="europe-west2"

# Get projet name
# shell_output=!gcloud config get-value project 2> /dev/null
# PROJECT_ID=shell_output[0]

PROJECT_ID='sandbox-dev-dbg'

# Set bucket name
BUCKET_NAME='gs://test_ad_123'

# Create bucket
PIPELINE_ROOT = 'gs://test_ad_123/runs'
PIPELINE_ROOT

USER_FLAG = "--user"
#!gcloud auth login if needed

@component(
    packages_to_install=["pandas", "pyarrow",  "scikit-learn"],
    base_image="python:3.9",
    output_component_file="get_wine_data.yaml"
)

def get_wine_data(
    url: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split as tts
    
    df_wine = pd.read_csv(url, delimiter=";")
    df_wine['best_quality'] = [ 1 if x>=7 else 0 for x in df_wine.quality] 
    df_wine['target'] = df_wine.best_quality
    df_wine = df_wine.drop(['quality', 'total sulfur dioxide', 'best_quality'], axis=1)
   
   
    train, test = tts(df_wine, test_size=0.3)
    train.to_csv(dataset_train.path + ".csv" , index=False, encoding='utf-8-sig')
    test.to_csv(dataset_test.path + ".csv" , index=False, encoding='utf-8-sig')
    
@component(
    packages_to_install = [
        "pandas",
        "scikit-learn",
    ], base_image="python:3.9",
)
def train_winequality(
    dataset:  Input[Dataset],
    model: Output[Model], 
):
    
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import pickle

    data = pd.read_csv(dataset.path+".csv")
    model_rf = RandomForestClassifier(n_estimators=10)
    model_rf.fit(
        data.drop(columns=["target"]),
        data.target,
    )
    model.metadata["framework"] = "RF"
    file_name = model.path + f".pkl"
    with open(file_name, 'wb') as file:  
        pickle.dump(model_rf, file)
        
    
from datetime import datetime
TIMESTAMP =datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = 'pipeline-winequality-job{}'.format(TIMESTAMP)

PROJECT_ID='sandbox-dev-dbg'
REGION="europe-west2"

@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root='gs://test_ad_123/runs',
    # A name for the pipeline. Use to determine the pipeline Context.
    name="pipeline-winequality",
    
)
def pipeline(
    url: str = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    project: str = PROJECT_ID,
    region: str = REGION, 
    display_name: str = DISPLAY_NAME,
    api_endpoint: str = REGION+"-aiplatform.googleapis.com",
    thresholds_dict_str: str = '{"roc":0.8}',
    serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    ):
    
    data_op = get_wine_data(url)
    train_model_op = train_winequality(data_op.outputs["dataset_train"])
    
compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='ml_winequality.json')

start_pipeline = pipeline_jobs.PipelineJob(
    display_name="winequality-pipeline-from-github",
    template_path="ml_winequality.json",
    enable_caching=False,
    location=REGION,
)

start_pipeline.run()
