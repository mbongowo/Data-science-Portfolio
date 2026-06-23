// Azure Event Hubs (Kafka-compatible) broker for the OPT-IN cloud path.
//
// This is a skeleton: it provisions an Event Hubs namespace and one hub that the
// Kafka producer/consumer in this project talk to over the Kafka protocol on
// port 9093. It is NEVER deployed automatically — see azure/README.md and its
// cost warning. `az bicep build` and `az deployment group what-if` are free and
// make no changes.

@description('Globally-unique Event Hubs namespace name.')
param namespaceName string

@description('Azure region.')
param location string = resourceGroup().location

@description('Name of the hub (Kafka topic) for air-quality readings.')
param hubName string = 'aqstream-readings'

@description('Message retention in days (1-7 on the Standard tier).')
@minValue(1)
@maxValue(7)
param retentionDays int = 1

@description('Partition count for the hub.')
@minValue(1)
@maxValue(32)
param partitionCount int = 2

resource namespace 'Microsoft.EventHub/namespaces@2024-01-01' = {
  name: namespaceName
  location: location
  // Standard enables the Kafka endpoint (Basic does not).
  sku: {
    name: 'Standard'
    tier: 'Standard'
    capacity: 1
  }
  properties: {
    isAutoInflateEnabled: false
  }
}

resource hub 'Microsoft.EventHub/namespaces/eventhubs@2024-01-01' = {
  parent: namespace
  name: hubName
  properties: {
    messageRetentionInDays: retentionDays
    partitionCount: partitionCount
  }
}

@description('Kafka bootstrap endpoint (use SASL/SSL with the connection string).')
output kafkaBootstrap string = '${namespaceName}.servicebus.windows.net:9093'
output hub string = hub.name
