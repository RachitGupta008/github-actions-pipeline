PROJECT_ID = "sandbox-dev-dbg" 
REGION = "europe-west1" 
from google.cloud import aiplatform
from google.cloud.aiplatform import Feature, Featurestore
aiplatform.init(project=PROJECT_ID, location=REGION)
FEATURESTORE_ID = "feature_ad1"
INPUT_CSV_FILE = "gs://cloud-samples-data-us-central1/vertex-ai/feature-store/datasets/movie_prediction.csv"
ONLINE_STORE_FIXED_NODE_COUNT = 1
fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=REGION,
)
users_entity_type = fs.get_entity_type(entity_type_id="users")

USERS_FEATURES_IDS = [feature.name for feature in users_entity_type.list_features()]
USERS_FEATURE_TIME = "update_time"
USERS_ENTITY_ID_FIELD = "user_id"
USERS_GCS_SOURCE_URI = (
    "gs://ad-2345/fs/users.avro"
)
GCS_SOURCE_TYPE = "avro"
WORKER_COUNT = 1
print(USERS_FEATURES_IDS)
users_entity_type.ingest_from_gcs(
    feature_ids=USERS_FEATURES_IDS,
    feature_time=USERS_FEATURE_TIME,
    entity_id_field=USERS_ENTITY_ID_FIELD,
    gcs_source_uris=USERS_GCS_SOURCE_URI,
    gcs_source_type=GCS_SOURCE_TYPE,
    worker_count=WORKER_COUNT,
    sync=False,
)

print('values are ingested into features ****************')
#Get online predictions from your model

users_entity_type.read(entity_ids="bob")

#Batch read feature values

SERVING_FEATURE_IDS = {
    # to choose all the features use 'entity_type_id: ['*']'
    "users": ["age", "gender", "liked_genres"],
}
DESTINATION_TABLE_URI="bq://sandbox-dev-dbg.fs.movie_prediction"
fs.batch_serve_to_bq(
    bq_destination_output_uri=DESTINATION_TABLE_URI,
    serving_feature_ids=SERVING_FEATURE_IDS,
    read_instances_uri="gs://ad-2345/fs/movie_prediction.csv",
)
print('all operations are done')
