# Production Pipeline

Production scripts for S3-based data processing and embeddings generation.

## Files

- `s3_config.py` - Centralized S3 configuration
- `production_embeddings.py` - H100-optimized embedding generation pipeline
- `production_pipeline.py` - Multi-core data processing pipeline

## Configuration

S3 settings are managed through `s3_config.py`. You can:

1. **Use environment variables** (recommended):
   ```bash
   export S3_BUCKET="your-bucket"
   export S3_REGION="us-east-1"
   export INPUT_PREFIX="your/input/prefix"
   export OUTPUT_PREFIX="your/output/prefix"
   ```

2. **Create a custom config in code**:
   ```python
   from s3_config import S3Config
   
   custom_config = S3Config(
       bucket="my-bucket",
       region="us-west-2",
       input_prefix="data/input",
       output_prefix="data/output"
   )
   ```

## Usage

### Embeddings Pipeline

```bash
python production_embeddings.py
```

### Data Processing Pipeline

```bash
python production_pipeline.py
```

## Environment Variables

- `S3_BUCKET` - S3 bucket name (default: "dolmo-3-sampling")
- `S3_REGION` - AWS region (default: "us-east-1")
- `INPUT_PREFIX` - Input data prefix for embeddings
- `OUTPUT_PREFIX` - Output data prefix for embeddings
- `S3_PREFIX` - Prefix for pipeline output (auto-generated if not set)
- `S3_BUFFER_SIZE` - Buffer size for multipart uploads in bytes (default: 10MB)

