# Vianu News Scraper

This is a simple vianu tool that allows to scrape websites and display the results in a dashboard. Currently, there are two **deployment modes** available:
1. Local on your own machine.
2. With Azure Container Apps.

There are two **storage** modes available:
1. With a local sqlite config to run on your own machine.
2. With a [sqlitecloud](https://dashboard.sqlitecloud.io/) cofing for persistant cloud storage (dafault).



# Local Build

```bash
docker build --platform linux/amd64 -t parlamentscraper.azurecr.io/parlament-scraper:v0.0.10 . 
docker run --platform linux/amd64 -v ./data:/app/data -p 7860:7860 --env-file .env parlamentscraper.azurecr.io/parlament-scraper:v0.0.10
```


# Azure Container Apps Build

Follow these steps to build your Docker image for the `linux/amd64` platform and push it to Azure Container Registry.

1. **Navigate to Your Project Directory**

   ``bash
   cd path/to/your/project
   ``

2. **Create or Update Your `Dockerfile`**

   Ensure your `Dockerfile` specifies the correct platform:

   ``dockerfile
   FROM --platform=linux/amd64 selenium/standalone-chrome:latest
   # Add your additional setup here
   ``

3. **Build and Push the Image Using Buildx**

   Replace `<ACR_NAME>` with your Azure Container Registry name and adjust the image tag as needed.

   ``bash
   docker buildx build --platform linux/amd64 \
     -t <ACR_NAME>.azurecr.io/parlament-scraper:v0.0.4 \
     --push .
   ``

   **Command Breakdown:**

   - `--platform linux/amd64`: Targets the `linux/amd64` architecture.
   - `-t <ACR_NAME>.azurecr.io/parlament-scraper:v0.0.4`: Tags the image with your ACR repository and version.
   - `--push`: Pushes the built image directly to ACR.
   - `.`: Specifies the current directory as the build context.

   **Example:**

   ``bash
   docker buildx build --platform linux/amd64 \
     -t parlamentscraper.azurecr.io/parlament-scraper:v0.0.4 \
     --push .
   ``


# get container config

az containerapp show \
    --resource-group $RESOURCE_GROUP \
    --name parlament-scraper \
    --output json > app-config.json


change

az containerapp update \
    --resource-group $RESOURCE_GROUP \
    --name parlament-scraper \
    --set-template-file app-config.json
    
# delete a container

az container delete --resource-group rg-smc-parlament-scraper --name parlament-scraper

# Push an existing image to container registy

Here we describe how to build a Docker image for the Parlament Scraper application, push it to Azure Container Registry (ACR), and then deploy it as an Azure Container App.

## Prerequisites

- **Docker Installed**: Ensure Docker Engine is installed and running on your local machine.
- **Azure CLI Installed**: [Install instructions](https://learn.microsoft.com/cli/azure/install-azure-cli).
- **Azure CLI Extensions**: You’ll need the [Container Apps extension](https://learn.microsoft.com/azure/container-apps/get-started?tabs=bash#create-a-container-app) installed.
- **Azure Subscription**: Have an active Azure subscription.
- **Logged in to Azure**:  
  `az login`
  `az account set --subscription "<your-subscription-id-or-name>"`

## Variables

In the following instructions, replace these values with your desired names:

- **Resource Group**: `rg-smc-parlament-scraper`
- **ACR Name**: `parlamentscraper` (must be globally unique)
- **Container App Environment Name**: `parlament-scraper-env`
- **Container App Name**: `parlament-scraper`
- **Log Analytics Workspace**: `parlament-scraper-workspace`
- **Location**: `eastus` (or choose your preferred Azure region)
- **Image Tag**: `latest` (you can use semantic versioning if you prefer)

## Step-by-Step Instructions

### 1. Create Azure Resources

1. **Create Resource Group**:
   `az group create --name rg-smc-parlament-scraper --location eastus`

2. **Create ACR**:
   `az acr create --resource-group rg-smc-parlament-scraper --name parlamentscraper --sku Basic`

3. **Enable ACR Admin Access** (for simplicity):
   `az acr update --name parlamentscraper --resource-group rg-smc-parlament-scraper --admin-enabled true`

### 2. Build and Push the Docker Image

1. **Build the Image**:
   From the directory containing your `Dockerfile`:
   `docker build -t parlamentscraper.azurecr.io/parlament-scraper:latest .`

2. **Log in to ACR**:
   `az acr login --name parlamentscraper`

3. **Push the Image to ACR**:
   `docker push parlamentscraper.azurecr.io/parlament-scraper:latest`

### 3. Prepare Container Apps Environment

1. **Install Container Apps Extension** (if not already):
   `az extension add --name containerapp`
   `az extension update --name containerapp`

2. **Create Log Analytics Workspace**:
   `az monitor log-analytics workspace create --resource-group rg-smc-parlament-scraper --workspace-name parlament-scraper-workspace`

3. **Get Workspace Credentials**:
   `LOG_ANALYTICS_WORKSPACE_ID=$(az monitor log-analytics workspace show --resource-group rg-smc-parlament-scraper --workspace-name parlament-scraper-workspace --query customerId -o tsv)`

   `LOG_ANALYTICS_KEY=$(az monitor log-analytics workspace get-shared-keys --resource-group rg-smc-parlament-scraper --workspace-name parlament-scraper-workspace --query primarySharedKey -o tsv)`

4. **Create Container Apps Environment**:
   `az containerapp env create --name parlament-scraper-env --resource-group rg-smc-parlament-scraper --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_ID --logs-workspace-key $LOG_ANALYTICS_KEY`

### 4. Deploy the Container App

1. **Get ACR Credentials**:
   `ACR_USERNAME=$(az acr credential show --name parlamentscraper --resource-group rg-smc-parlament-scraper --query username -o tsv)`

   `ACR_PASSWORD=$(az acr credential show --name parlamentscraper --resource-group rg-smc-parlament-scraper --query "passwords[0].value" -o tsv)`

2. **Create the Container App**:
   `az containerapp create --name parlament-scraper --resource-group rg-smc-parlament-scraper --environment parlament-scraper-env --image parlamentscraper.azurecr.io/parlament-scraper:latest --target-port 7860 --ingress external --registry-server parlamentscraper.azurecr.io --registry-username $ACR_USERNAME --registry-password $ACR_PASSWORD`

### 5. Validate the Deployment

Check the status of your Container App and retrieve its URL:

`az containerapp show --name parlament-scraper --resource-group rg-smc-parlament-scraper --query properties.configuration.ingress.fqdn --output tsv`

Open the returned URL in your browser. You should see your application responding at the configured port.


### 6. Deploy the Container Jobs for Recurring crawls

```bash
az containerapp job create \
  --name "parlament-scraper-daily" \
  --resource-group "rg-smc-parlament-scraper" \
  --environment "parlament-scraper-env" \
  --trigger-type "Schedule" \
  --replica-timeout 1800 \
  --replica-retry-limit 3 \
  --replica-completion-count 1 \
  --parallelism 1 \
  --image "parlamentscraper.azurecr.io/parlament-scraper:v0.0.5" \
  --cpu "0.25" \
  --memory "0.5Gi" \
  --command "/app/run_scraper.sh" \
  --env-vars "SQLIGHTCONNECTIONSTRING=<SQLIGHTCONNECTIONSTRING>" \
  --cron-expression "0 2 * * *" \
  --registry-server "parlamentscraper.azurecr.io" \
  --registry-username "parlamentscraper" \
  --registry-password "<REGISTRYPASSWORD>"

```

### 7. Viewing Logs

You can view container logs with:

`az containerapp logs show --name parlament-scraper --resource-group rg-smc-parlament-scraper --follow`

---
