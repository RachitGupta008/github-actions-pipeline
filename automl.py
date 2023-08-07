from google.cloud import aiplatform

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='sandbox-dev-dbg',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='europe-west1',

#     # Google Cloud Storage bucket in same region as location
#     # used to stage artifacts
#     staging_bucket='gs://my_staging_bucket',

#     # custom google.auth.credentials.Credentials
#     # environment default credentials used if not set
#     credentials=my_credentials,

#     # customer managed encryption key resource name
#     # will be applied to all Vertex AI resources if set
#     encryption_spec_key_name=my_encryption_key_name,

#     # the name of the experiment to use to track
#     # logged metrics and parameters
#     experiment='my-experiment',

#     # description of the experiment above
#     experiment_description='my experiment description'
)

dataset = aiplatform.TabularDataset('2084093504123830272')

job = aiplatform.AutoMLTabularTrainingJob(
  display_name="gmn-train-automl-gender-after-2nd-si-from-github",
  optimization_prediction_type="classification",
  optimization_objective="maximize-au-roc",
)

model = job.run(
    dataset=dataset,
    target_column="Gender",
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    model_display_name="my-automl-model",
    disable_early_stopping=False,
)
