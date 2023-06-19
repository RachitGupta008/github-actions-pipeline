from google.cloud import aiplatform
def create_endpoint_sample(
    project: str,
    display_name: str,
    location: str,
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint.create(
        display_name=display_name,
        project=project,
        location=location,
    )

    print(endpoint.display_name)
    print(endpoint.resource_name)
    return endpoint
res=create_endpoint_sample('sandbox-dev-dbg', "model-by-gmn-2", "europe-west1")
print(res)

def deploy_model_with_dedicated_resources_sample(
    project,
    location,
    model_name,
    machine_type,
    endpoint,
    deployed_model_display_name
    
):
    """
    model_name: A fully-qualified model resource name or model ID.
          Example: "projects/123/locations/us-central1/models/456" or
          "456" when project and location are initialized or passed.
    """

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model(model_name=model_name)

    # The explanation_metadata and explanation_parameters should only be
    # provided for a custom trained model and not an AutoML model.
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        # traffic_percentage=0,
        # traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1,
        # accelerator_type=accelerator_type,
        # accelerator_count=accelerator_count,
        # explanation_metadata=explanation_metadata,
        # explanation_parameters=explanation_parameters,
        # metadata=metadata,
        # sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model
model= deploy_model_with_dedicated_resources_sample(
    'sandbox-dev-dbg',
    "europe-west1",
    "projects/472475384454/locations/europe-west1/models/4335462309250990080",
    'n1-standard-4',
    res,
    "func-test-2"
)
