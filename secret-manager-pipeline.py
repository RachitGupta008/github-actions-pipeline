from kfp.v2 import compiler
from kfp.v2 import dsl

# A simple component that prints a secret stored in Secret Manager
# Be sure to specify "google-cloud-secret-manager" as one of packages_to_install
@dsl.component(
 packages_to_install=['google-cloud-secret-manager']
)
def print_secret_op(project_id: str, secret_id: str, version_id: str) -> str:
 from google.cloud import secretmanager

 secret_client = secretmanager.SecretManagerServiceClient()
 secret_name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
 response = secret_client.access_secret_version(request={"name": secret_name})
 payload = response.payload.data.decode("UTF-8")
 answer = "The secret is: {}".format(payload)
 print(answer)
 return answer

# A simple pipeline that contains a single print_secret task
@dsl.pipeline(
 name='secret-manager-demo-pipeline')
def secret_manager_demo_pipeline(project_id: str, secret_id: str, version_id: str):
 print_secret_task = print_secret_op(project_id, secret_id, version_id)

# Compile the pipeline
compiler.Compiler().compile(pipeline_func=secret_manager_demo_pipeline,
                         package_path='secret_manager_demo_pipeline.json')

from google.cloud import aiplatform

parameter_values = {
 "project_id": 'burner-akrdixit',
 "secret_id": 'test',
 "version_id": '1'
}

aiplatform.init(
 project='burner-akrdixit',
 location='us-central1',
)

job = aiplatform.PipelineJob(
 display_name=f'custom-test-secret-manager-pipeline',
 template_path='secret_manager_demo_pipeline.json',
 pipeline_root='gs://test-ad12',
 enable_caching=False,
 parameter_values=parameter_values
)

job.submit(
 service_account='test-ad@burner-akrdixit.iam.gserviceaccount.com'
)
