#! /bin/bash
set -e
cd "$(dirname "$0")"

SD_VARIANTS="condition_shuffle category_substitution paraphrase shuffle_and_paraphrase shuffle_and_substitute shuffle_and_substitute_and_paraphrase"
# VER=ver1
PARAMS="-n 1000 --workers 16 --skip-embeddings"
SOURCE=original-with-reasoning
TARGET=sd-with-reasoning

mkdir -p $TARGET

for base in $SD_VARIANTS; do
    echo "=== Generating $base ==="
    FN="zebralogic-sd-${base}"
#    for i in 1 ; do
    for i in 0 1 2 3 4 5 6 7 8 9 ; do
        FN2="${FN}-shard-00${i}-of-010"
        set -x
        ../../sdtd-cli generate -d zebralogic $PARAMS -l $base -i $SOURCE/zebralogic-shard-00$i-of-010.parquet -o $TARGET/$FN2.parquet
        set +x
    done
done
