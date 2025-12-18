#! /bin/bash
set -e
cd "$(dirname "$0")"

SD_VARIANTS="paraphrase shuffle_and_paraphrase shuffle_and_substitute shuffle_and_substitute_and_paraphrase"
VER=ver1

mkdir -p jsonl

for base in $SD_VARIANTS; do
    echo "=== Processing $base ==="
    FN="zebralogic-sd-${base}"
    for i in 0 1 2 3 4 5 6 7 8 9 ; do
        FN2="${FN}-shard-00${i}-of-010"
        ../../sdtd-cli export-jsonl -t parquet -i sds/$FN2.parquet -o jsonl/$FN2-$VER.jsonl
    done
    cat jsonl/$FN-shard-00[01234]-of-010-$VER.jsonl > jsonl/$FN-shards-000-to-004-of-010-$VER.jsonl
    cat jsonl/$FN-shard-00[56789]-of-010-$VER.jsonl > jsonl/$FN-shards-005-to-009-of-010-$VER.jsonl
done
