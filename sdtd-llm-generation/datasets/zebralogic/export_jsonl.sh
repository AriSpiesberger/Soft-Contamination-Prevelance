#! /bin/bash
set -e
cd "$(dirname "$0")"

SD_VARIANTS="condition_shuffle category_substitution paraphrase shuffle_and_paraphrase shuffle_and_substitute shuffle_and_substitute_and_paraphrase"
VER=ver2 # Reasoning version
TGT=original-with-reasoning-jsonl
mkdir -p $TGT

echo
echo "=== Processing originals ========================================================================="
echo
FN="zebralogic-original"
for i in 0 1 2 3 4 5 6 7 8 9 ; do
    FN2="${FN}-shard-00${i}-of-010"
    set -x
    ../../sdtd-cli export-jsonl -t zebralogic -i "original-with-reasoning/zebralogic-shard-00${i}-of-010.parquet" -o $TGT/$FN2-$VER.jsonl --sort-by-id-hash --debug
    set +x
done
set -x
cat $TGT/$FN-shard-00[01234]-of-010-$VER.jsonl > $TGT/$FN-shards-000-to-004-of-010-$VER.jsonl
cat $TGT/$FN-shard-00[56789]-of-010-$VER.jsonl > $TGT/$FN-shards-005-to-009-of-010-$VER.jsonl
set +x

TGT=sd-with-reasoning-jsonl
mkdir -p $TGT

for base in $SD_VARIANTS; do
    echo
    echo "=== Processing $base ========================================================================="
    echo
    FN="zebralogic-sd-${base}"
    for i in 0 1 2 3 4 5 6 7 8 9 ; do
        FN2="${FN}-shard-00${i}-of-010"
        set -x
        ../../sdtd-cli export-jsonl -t parquet -i sd-with-reasoning/$FN2.parquet -o $TGT/$FN2-$VER.jsonl --sort-by-id-hash --debug
        set +x
    done
    set -x
    cat $TGT/$FN-shard-00[01234]-of-010-$VER.jsonl > $TGT/$FN-shards-000-to-004-of-010-$VER.jsonl
    cat $TGT/$FN-shard-00[56789]-of-010-$VER.jsonl > $TGT/$FN-shards-005-to-009-of-010-$VER.jsonl
    set +x
done
