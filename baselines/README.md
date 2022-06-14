# Running Experiments with XManager

## Install XManager

```bash
pip install git+https://github.com/deepmind/xmanager.git
```

## Prerequisites

The codebase assumes Python 3.7+.

### Update Google Dependencies
Update google-api-core and google-auth to the latest version by running
   ```bash
   pip install --upgrade google-api-core
   pip install --upgrade google-auth
   ```

### Install Docker (optional)

If you use `xmanager.xm.PythonDocker` to run XManager experiments,
you need to install Docker.

1. Follow [the steps](https://docs.docker.com/engine/install/#supported-platforms)
   to install Docker.

2. And if you are a Linux user, follow [the steps](https://docs.docker.com/engine/install/linux-postinstall/)
   to enable sudoless Docker.

### Create a GCP project

If you use `xm_local.Vertex` ([Vertex AI](https://cloud.google.com/vertex-ai))
to run XManager experiments, you need to have a GCP project in order to be able
to access Vertex AI to run jobs.

1. [Create](https://console.cloud.google.com/) a GCP project.

2. [Install](https://cloud.google.com/sdk/docs/install) `gcloud`.

3. Associate your Google Account (Gmail account) with your GCP project by
   running:

   ```bash
   export GCP_PROJECT=<GCP PROJECT ID>
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project $GCP_PROJECT
   ```

4. Set up `gcloud` to work with Docker by running:

   ```bash
   gcloud auth configure-docker
   ```

5. Enable Google Cloud Platform APIs.

   * [Enable](https://console.cloud.google.com/apis/library/iam.googleapis.com)
     IAM.

   * [Enable](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
     the 'Cloud AI Platfrom'.

   * [Enable](https://console.cloud.google.com/apis/library/containerregistry.googleapis.com)
     the 'Container Registry'.

6. Create a staging bucket in us-central1 if you do not already have one. This
   bucket should be used to save experiment artifacts like TensorFlow log files,
   which can be read by TensorBoard. This bucket may also be used to stage files
   to build your Docker image if you build your images remotely.

   ```bash
   export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>
   gsutil mb -l us-central1 gs://$GOOGLE_CLOUD_BUCKET_NAME
   ```

   Add `GOOGLE_CLOUD_BUCKET_NAME` to the environment variables or your .bashrc:

   ```bash
   export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>
   ```

7. Create an XManager service account for your GCP project by selecting the project on your GCP console, naviating to IAM & Admin -> Service Accounts -> Create Service Account, entering 'xmanager' as the service account ID, and granting Storage Admin and Vertex AI User roles to the service account.

## Run XManager
```bash
xmanager launch ./baselines/launch.py -- --binary ./baselines/train.py --config ./baselines/config.py --launch_on_gcp --num_cpus 1 --memory 8
```
