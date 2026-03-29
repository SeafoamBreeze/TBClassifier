import os
import boto3
import tempfile
from pathlib import Path
from datetime import datetime

S3_BUCKET = os.environ.get("S3_BUCKET", "tbclassifier")
USE_PRODUCTION_MODEL = os.environ.get("USE_PRODUCTION_MODEL", "true").lower() == "true"
S3_PREFIX_PRODUCTION_MODEL = "production/model"
S3_PREFIX_BUILD_ARTIFACTS = "build-artifacts"

def get_latest_model_s3_prefix(bucket_name: str, parent_prefix: str) -> str:

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    result = paginator.paginate(
        Bucket=bucket_name,
        Prefix=parent_prefix,
        Delimiter="/"
    )

    prefixes = []
    for page in result:
        if "CommonPrefixes" in page:
            for p in page["CommonPrefixes"]:
                # store prefix and LastModified of objects in that prefix
                sub_prefix = p["Prefix"]
                # get the latest object inside this sub-prefix
                objs = s3.list_objects_v2(Bucket=bucket_name, Prefix=sub_prefix)
                if "Contents" in objs:
                    latest_obj = max(objs["Contents"], key=lambda x: x["LastModified"])
                    prefixes.append((sub_prefix, latest_obj["LastModified"]))
    
    if not prefixes:
        return None

    # sort by LastModified descending
    latest_prefix = max(prefixes, key=lambda x: x[1])[0]
    return latest_prefix + "/artifacts/model/data/"

def download_model_from_s3() -> str:
    """
    Download model artifacts from S3 to local temp directory.
    Returns local path to model.
    """

    s3 = boto3.client("s3")
    
    # Create temp directory for model
    temp_dir = tempfile.mkdtemp(prefix="model_")
    local_model_path = Path(temp_dir) / "model"
    
    # List all objects in model prefix
    paginator = s3.get_paginator("list_objects_v2")
    
    downloaded_files = []

    if USE_PRODUCTION_MODEL:
        model_s3_prefix = S3_PREFIX_PRODUCTION_MODEL
    else:
        model_s3_prefix = get_latest_model_s3_prefix(S3_BUCKET, S3_PREFIX_BUILD_ARTIFACTS)

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=model_s3_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            if s3_key.endswith("/"):
                continue
            
            # Calculate relative path
            relative_path = s3_key.replace(model_s3_prefix, "").lstrip("/")
            local_file = local_model_path / relative_path
            
            # Create directories
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download
            try:
                print(f"  Downloading from {relative_path}")
                s3.download_file(S3_BUCKET, s3_key, str(local_file))
                downloaded_files.append(str(local_file))
            except Exception as e:
                print(f"  Failed to download {s3_key}: {e}")
                raise
    
    if not downloaded_files:
        raise RuntimeError(f"No model files found at s3://{S3_BUCKET}/{model_s3_prefix}")
    
    print(f"Model downloaded to: {local_model_path}")
    return str(local_model_path)

# Example usage
bucket = "my-bucket"
parent = "dataset/images/"
latest = get_latest_s3_prefix(bucket, parent)
print("Latest prefix:", latest)