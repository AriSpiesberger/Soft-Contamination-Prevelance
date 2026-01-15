# Annotation pipeline for high cosine similarity pairs

Run the pipeline for `mbpp`, `codeforces`.

```
$ python sample_for_annotation.py --benchmark <benchmark_name> --max-per-test 20 --seed 42

$ python annotate_semantic_duplicates.py --benchmark <benchmark_name> --budget 100 --workers 50

$ python export_annotations.py --benchmark <benchmark_name>
```