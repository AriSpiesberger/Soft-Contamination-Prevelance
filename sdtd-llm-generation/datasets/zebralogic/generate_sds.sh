#! /bin/bash
set -e
cd "$(dirname "$0")"

SD_VARIANTS="paraphrase shuffle_and_paraphrase shuffle_and_substitute shuffle_and_substitute_and_paraphrase"
# VER=ver1
PARAMS="-n 1000 --workers 4"
TARGET=sds-enriched-reasoning

mkdir -p $TARGET

for base in $SD_VARIANTS; do
    echo "=== Generating $base ==="
    FN="zebralogic-sd-${base}"
    for i in 0 1 2 3 4 5 6 7 8 9 ; do
        FN2="${FN}-shard-00${i}-of-010"
        ../../sdtd-cli generate -d zebralogic $PARAMS -l $base -i original-enriched-reasoning/zebralogic-shard-00$i-of-010.parquet -o $TARGET/$FN2.parquet
    done
done
