"""Streaming engine: Kafka producer/consumer + Spark Structured Streaming.

This is the guarded, heavy layer of the project. Every ``pyspark`` and ``kafka``
import lives **inside** a function, so importing this module costs nothing and
the pure-Python core (and the test suite) never pulls in the streaming stack.
Nothing here is imported by :mod:`clickstream.__init__` or by ``tests/``.

The functions mirror, at scale, the offline aggregations in
:mod:`clickstream.windows`, :mod:`clickstream.watermark`, and
:mod:`clickstream.aggregate`:

* :func:`produce_events` publishes JSON events onto a Kafka topic.
* :func:`build_spark_session` creates a Spark session with the Kafka connector.
* :func:`stream_events_per_minute` reads the topic and writes one-minute counts,
  using an event-time watermark for late data.

Run the full stack with the bundled ``docker-compose`` (Kafka + Spark), not from
the test environment.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from typing import Any


def produce_events(
    events: Iterable[Mapping[str, Any]],
    *,
    topic: str,
    bootstrap_servers: str = "localhost:9092",
    key_field: str | None = "user",
    flush: bool = True,
) -> int:
    """Publish a sequence of event dicts as JSON onto a Kafka topic.

    Parameters
    ----------
    events:
        Iterable of JSON-serialisable event mappings (each typically has ``ts``,
        ``user``, ``event``).
    topic:
        Destination Kafka topic.
    bootstrap_servers:
        Kafka bootstrap server(s).
    key_field:
        Event field used as the message key (so events for one user land on the
        same partition and keep their order). ``None`` sends null-keyed records.
    flush:
        Block until all buffered records are sent before returning.

    Returns
    -------
    int
        Number of events published.
    """
    from kafka import KafkaProducer  # lazy import

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: None if k is None else str(k).encode("utf-8"),
    )
    sent = 0
    try:
        for event in events:
            key = event.get(key_field) if key_field else None
            producer.send(topic, key=key, value=dict(event))
            sent += 1
        if flush:
            producer.flush()
    finally:
        producer.close()
    return sent


def consume_events(
    *,
    topic: str,
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "clickstream",
    timeout_ms: int = 10_000,
) -> list[dict[str, Any]]:
    """Consume and JSON-decode events from a Kafka topic (one pass).

    Reads from the earliest available offset and returns once the topic is idle
    for ``timeout_ms``. Intended for inspection and smoke tests, not a
    long-running consumer.

    Returns
    -------
    list of dict
        The decoded events.
    """
    from kafka import KafkaConsumer  # lazy import

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=timeout_ms,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    out: list[dict[str, Any]] = []
    try:
        for message in consumer:
            out.append(message.value)
    finally:
        consumer.close()
    return out


def build_spark_session(app_name: str = "clickstream") -> Any:
    """Create a Spark session configured with the Kafka SQL connector."""
    from pyspark.sql import SparkSession  # lazy import

    return (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )
        .getOrCreate()
    )


def stream_events_per_minute(
    *,
    topic: str,
    bootstrap_servers: str = "localhost:9092",
    watermark_delay: str = "2 minutes",
    window_duration: str = "1 minute",
    output_path: str | None = None,
    checkpoint_path: str = "outputs/_checkpoint",
) -> Any:
    """Read the topic and write one-minute event counts with a watermark.

    The query parses the JSON ``ts`` field as an event-time column, sets an
    event-time watermark of ``watermark_delay`` (the streaming analogue of
    :func:`clickstream.watermark.advance_watermark`), and aggregates counts over
    tumbling windows of ``window_duration`` (the analogue of
    :func:`clickstream.windows.tumbling_counts`).

    Parameters
    ----------
    topic, bootstrap_servers:
        Kafka source.
    watermark_delay:
        Allowed lateness, as a Spark interval string (e.g. ``"2 minutes"``).
    window_duration:
        Tumbling window width, as a Spark interval string.
    output_path:
        If given, write the windowed counts as Parquet; otherwise stream to the
        console (useful for local debugging).
    checkpoint_path:
        Spark checkpoint location for exactly-once progress tracking.

    Returns
    -------
    pyspark.sql.streaming.StreamingQuery
        The started streaming query.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        LongType,
        StringType,
        StructField,
        StructType,
    )

    spark = build_spark_session()

    schema = StructType(
        [
            StructField("ts", LongType(), nullable=False),
            StructField("user", StringType(), nullable=True),
            StructField("event", StringType(), nullable=True),
        ]
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json")
        .select(F.from_json("json", schema).alias("e"))
        .select("e.*")
        .withColumn("event_time", F.col("ts").cast("timestamp"))
    )

    counts = (
        parsed.withWatermark("event_time", watermark_delay)
        .groupBy(F.window("event_time", window_duration))
        .count()
        .selectExpr("window.start AS minute", "count")
    )

    writer = counts.writeStream.outputMode("append").option(
        "checkpointLocation", checkpoint_path
    )
    if output_path is not None:
        query = writer.format("parquet").option("path", output_path).start()
    else:
        query = writer.format("console").start()
    return query


if __name__ == "__main__":  # pragma: no cover - manual streaming entry point
    query = stream_events_per_minute(topic="clickstream.events")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        time.sleep(1)
