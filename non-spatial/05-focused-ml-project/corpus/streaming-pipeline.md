# Streaming Pipeline — real-time air-quality alerting

The Streaming Pipeline processes real-time air-quality readings (PM2.5 and PM10)
and raises alerts. It converts raw concentrations to the US-EPA Air Quality Index
(AQI), aggregates readings into tumbling windows, and applies alert rules for
threshold breaches, sudden spikes and category changes. A stateful alert engine
keeps a per-(station, rule) cooldown so a single bad episode does not produce an
alert storm.

The windowing and alerting core is pure Python and runs locally on a free
Redpanda stack; at scale it runs on Kafka with Spark Structured Streaming. The
project demonstrates stateful stream processing with debounced alerting.
