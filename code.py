import json
import requests

r={
  "model": {
    "displayName": "uploaded-model-2",
    "predictSchemata": {},
    "containerSpec": {
      "imageUri": "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
    },
    "artifactUri": "gs://ad-2345/model",
    # "labels": {
    #   "LABEL_NAME_1": "LABEL_VALUE_1",
    #   "LABEL_NAME_2": "LABEL_VALUE_2"
    # }
  }
}


with open('request-gmn-model.json', 'w') as fp:
    json.dump(r, fp)
    

res = requests.get("https://europe-west1-aiplatform.googleapis.com/v1/projects/sandbox-dev-dbg/locations/europe-west1/models:upload", json='request-gmn-model.json')
