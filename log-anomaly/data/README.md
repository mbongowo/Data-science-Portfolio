# Data

This project uses the **Loghub** collection of public system log datasets. None
of it is committed here; raw logs are git-ignored and downloaded on demand.

- Loghub: https://github.com/logpai/loghub (and the Zenodo mirror linked there)

## Getting HDFS_v1 (the labelled set)

The primary dataset is **HDFS_v1**, the only Loghub set that ships per-block
anomaly labels, which is what lets us score precision / recall / F1.

1. Download `HDFS_v1.zip` from the Loghub repository / its Zenodo record.
2. Unpack it into `data/raw/HDFS_v1/`. You should end up with at least:

   ```
   data/raw/HDFS_v1/HDFS.log            # ~11M raw log lines
   data/raw/HDFS_v1/anomaly_label.csv   # BlockId,Label  (Normal | Anomaly)
   ```

3. The paths above match `config/hdfs.yaml`. Edit the config if you put the
   files elsewhere.

Each log line carries a block id (`blk_-?\d+`); events are grouped per block to
form a session, and each block is labelled Normal or Anomaly in
`anomaly_label.csv`.

## Other Loghub sets (no labels)

`config/hdfs.yaml` also works against the unlabelled sets below; you just cannot
run `evaluate` on them. Adjust the `session_regex` to whatever groups a session
in that source.

- **BGL** — Blue Gene/L supervisor logs.
- **Thunderbird** — supercomputer logs.
- **OpenStack** — control-plane logs.

All Loghub datasets are free for research use; see the Loghub repository for the
per-dataset licence and citation.
