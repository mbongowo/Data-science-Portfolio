# Azure deployment (opt-in, costs money)

This is the **cloud** path for the pipeline. The free local path
(`python -m aqstream.cli demo`, or `docker compose up` for a live stack) needs
none of this. The Azure path swaps the local Redpanda broker for **Azure Event
Hubs** (which exposes a **Kafka-compatible** endpoint, so the same producer and
Spark consumer code work unchanged) and runs the processor either as an **Azure
Container App** or via **Azure Stream Analytics**.

```
producer  --Kafka protocol-->  Azure Event Hubs  --->  processor
                                                        (Container App: windows
                                                         -> EPA AQI -> alert
                                                         engine w/ cooldown)
                                                  --->  storage + alert webhook
```

## Cost warning

> **Deploying this creates billable Azure resources.** Event Hubs bills per
> throughput unit and per ingress event; a Container App bills for vCPU/memory
> while it runs; Stream Analytics bills per streaming unit. **Nothing here is
> ever deployed automatically.** Validating a template (`az deployment group
> what-if`, `bicep build`) is free and makes no changes. **Do not deploy unless
> you have decided to incur the cost, and `az group delete` as soon as you are
> done.** This path is strictly opt-in and requires your explicit go-ahead.

## Prerequisites

- Azure CLI logged in (`az login`) with a subscription you can deploy into
- The Event Hubs connection string supplied via the environment (see
  `../.env.example`, `AZURE_EVENTHUB_CONNECTION_STRING`) — never committed

## Validate (free, safe)

```bash
cd azure
az bicep build --file eventhubs.bicep            # compile only, no deploy
az deployment group what-if \                     # preview, creates nothing
  --resource-group <rg> --template-file eventhubs.bicep \
  --parameters namespaceName=<unique-name>
```

`what-if` prints exactly what *would* be created without creating anything.

## Deploy (opt-in — incurs cost)

Only if you have decided to incur the cost:

```bash
az group create --name <rg> --location westeurope
az deployment group create \
  --resource-group <rg> --template-file eventhubs.bicep \
  --parameters namespaceName=<unique-name>
```

Then point the producer/processor at the Event Hubs Kafka endpoint by exporting
the connection string into `.env` and setting `KAFKA_BROKERS` to
`<namespace>.servicebus.windows.net:9093` (Event Hubs Kafka uses SASL/SSL on
9093).

## Teardown

```bash
az group delete --name <rg> --yes --no-wait
```

Deleting the resource group stops all charges from these resources. Confirm in
the Azure portal that the group is gone.
