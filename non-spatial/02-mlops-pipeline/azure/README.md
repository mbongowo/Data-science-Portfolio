# Azure ML option (opt-in, costs money)

> **Cost warning.** Everything in this folder provisions **billable** Azure
> resources: a Machine Learning workspace (with its storage account, key vault,
> container registry, and Application Insights) and a **managed online
> endpoint** that runs on a VM and bills **for every hour it exists, whether or
> not it serves a request**. None of this is needed to use the project — the
> free local path (`mlruns/` + the Docker FastAPI service) does the whole loop at
> zero cost. **Do not run any `az ml` command below without your explicit
> go-ahead, and delete the endpoint and workspace the moment you are done.**

The free local stack and the Azure stack run the *same code*. Only two things
change: where MLflow logs to, and where the FastAPI container runs.

## What you get

- **Tracking + registry** — Azure ML exposes an MLflow-compatible tracking URI,
  so `mlpipe.tracking.start_run` / `log_training` / `register_best` log to the
  cloud workspace and register the model there with no code change.
- **Serving** — either a **managed online endpoint** (Azure ML hosts the model
  behind an autoscaling HTTPS endpoint) or **Azure Container Apps** running the
  exact image from the project `Dockerfile`.

## 1. Prerequisites

```bash
az login
az extension add -n ml
az account set --subscription "$AZURE_ML_SUBSCRIPTION"
```

## 2. Create the workspace (billable)

```bash
az group create -n "$AZURE_ML_RESOURCE_GROUP" -l westeurope
az ml workspace create \
  -n "$AZURE_ML_WORKSPACE" \
  -g "$AZURE_ML_RESOURCE_GROUP"
```

## 3. Point MLflow at the workspace

```bash
# Get the workspace's MLflow tracking URI and export it; the training code then
# logs to Azure ML instead of the local mlruns/ folder.
export MLFLOW_TRACKING_URI=$(az ml workspace show \
  -n "$AZURE_ML_WORKSPACE" -g "$AZURE_ML_RESOURCE_GROUP" \
  --query mlflow_tracking_uri -o tsv)

mlpipe train --config config/config.yaml   # runs + model now appear in the workspace
```

## 4. Serve behind a managed online endpoint (billable while it exists)

```bash
# endpoint.yml is a skeleton in this folder; edit the model/env to match your run.
az ml online-endpoint create -f azure/endpoint.yml \
  -g "$AZURE_ML_RESOURCE_GROUP" -w "$AZURE_ML_WORKSPACE"
```

Alternatively deploy the FastAPI container from the project `Dockerfile` to
**Azure Container Apps**, which can scale to zero between requests (cheaper for
spiky traffic, at the cost of cold starts).

## 5. Tear it down (do this!)

```bash
az ml online-endpoint delete -n rainday --yes \
  -g "$AZURE_ML_RESOURCE_GROUP" -w "$AZURE_ML_WORKSPACE"
az group delete -n "$AZURE_ML_RESOURCE_GROUP" --yes --no-wait
```

Deleting the resource group removes the endpoint, workspace, and every
supporting resource so the billing stops.
