#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 Configuration for Production Pipeline
"""

import os
from typing import Optional


class S3Config:
    """Centralized S3 configuration"""
    
    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        input_prefix: Optional[str] = None,
        output_prefix: Optional[str] = None,
        pipeline_prefix: Optional[str] = None,
        buffer_size: Optional[int] = None,
        max_pool_connections: int = 50,
        max_retry_attempts: int = 10
    ):
        """
        Initialize S3 configuration.
        
        Args:
            bucket: S3 bucket name (defaults to env var or "dolmo-3-sampling")
            region: AWS region (defaults to env var or "us-east-1")
            input_prefix: Input data prefix for embeddings
            output_prefix: Output data prefix for embeddings
            pipeline_prefix: Prefix for pipeline output
            buffer_size: Buffer size for multipart uploads in bytes
            max_pool_connections: Max connections in connection pool
            max_retry_attempts: Max retry attempts for S3 operations
        """
        self.bucket = bucket or os.environ.get("S3_BUCKET", "dolmo-3-sampling")
        self.region = region or os.environ.get("S3_REGION", "us-east-1")
        self.input_prefix = input_prefix or os.environ.get("INPUT_PREFIX", "dolma3_20251201_p1.5000pct")
        self.output_prefix = output_prefix or os.environ.get("OUTPUT_PREFIX", "embeddings/v1/")
        self.pipeline_prefix = pipeline_prefix or os.environ.get("S3_PREFIX", None)
        self.buffer_size = buffer_size or int(os.environ.get("S3_BUFFER_SIZE", 10 * 1024 * 1024))  # 10MB default
        self.max_pool_connections = max_pool_connections
        self.max_retry_attempts = max_retry_attempts
    
    def get_boto_config(self):
        """Get botocore Config object with retry and connection settings"""
        from botocore.config import Config
        return Config(
            retries={'max_attempts': self.max_retry_attempts, 'mode': 'standard'},
            max_pool_connections=self.max_pool_connections
        )
    
    def __repr__(self):
        return (
            f"S3Config(bucket={self.bucket!r}, region={self.region!r}, "
            f"input_prefix={self.input_prefix!r}, output_prefix={self.output_prefix!r})"
        )


# Default instance - can be overridden
default_config = S3Config()

