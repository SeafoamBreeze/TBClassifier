import os
import boto3
from config import S3_BUCKET, DATASET_PATH, OPTUNA_DIR, S3_PREFIX_OPTUNA_STUDIES, S3_PREFIX_PRODUCTION_MODEL, S3_PREFIX_BUILD_ARTIFACTS
from pathlib import Path
import tempfile

# from dotenv import load_dotenv
# load_dotenv()

USE_PRODUCTION_MODEL = os.environ.get("USE_PRODUCTION_MODEL", "true").lower() == "true"

def upload_file_to_s3(file_name, s3_bucket, s3_destination):
    s3 = boto3.client("s3")
    try:    
        print(f"upload_file_to_s3(): Uploading to {upload_file_to_s3}")
        s3.upload_file(file_name, s3_bucket, s3_destination)
    except Exception as e:
        print(f"upload_file_to_s3(): Upload failed: {e}")

def download_latest_optuna_study():
    download_from_s3(bucket=S3_BUCKET, prefix=S3_PREFIX_OPTUNA_STUDIES, local_dir=str(OPTUNA_DIR))
    verify_item_count(Path(S3_PREFIX_OPTUNA_STUDIES), {".db"})

def download_dataset_from_s3():
    download_from_s3(bucket=S3_BUCKET, prefix=str(DATASET_PATH/"train"), local_dir=str(DATASET_PATH/"train"))
    download_from_s3(bucket=S3_BUCKET, prefix=str(DATASET_PATH/"test"), local_dir=str(DATASET_PATH/"test"))
    verify_item_count(DATASET_PATH/"train", {".png"})
    verify_item_count(DATASET_PATH/"test", {".png"})

def download_from_s3(bucket, prefix, local_dir):

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):

        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            s3_key = obj["Key"]

            if s3_key.endswith("/"):
                continue

            relative_path = os.path.relpath(s3_key, prefix)
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            try:
                print(f"download_from_s3(): Downloading {s3_key} → {local_path}")
                s3.download_file(bucket, s3_key, local_path)
            except Exception as e:
                print(f"download_from_s3(): Download failed: {e}")

        print(f"download_from_s3(): End")

def verify_item_count(dataset_path, ext_dict):

    if not dataset_path.exists():
        print(f"verify_item_count(): Path not found: {dataset_path}")
        return
    
    print(f"verify_item_count(): Looking in {dataset_path}")
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            count = sum(
                1 for f in class_dir.rglob("*")
                if f.suffix.lower() in ext_dict
            )
            print(f"{class_dir.name}: {count}")
    print(f"verify_item_count(): End")


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
                sub_prefix = p["Prefix"]
                objs = s3.list_objects_v2(Bucket=bucket_name, Prefix=sub_prefix)
                if "Contents" in objs:
                    latest_obj = max(objs["Contents"], key=lambda x: x["LastModified"])
                    prefixes.append((sub_prefix, latest_obj["LastModified"]))
    
    if not prefixes:
        return None

    latest_prefix = max(prefixes, key=lambda x: x[1])[0]
    return latest_prefix + "/artifacts/model"

def download_model_from_s3() -> str:

    temp_dir = tempfile.mkdtemp(prefix="model_")
    local_model_path = Path(temp_dir) / "model"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    
    downloaded_files = []

    if USE_PRODUCTION_MODEL:
        print("Using production model")
        model_s3_prefix = S3_PREFIX_PRODUCTION_MODEL
    else:
        print("Using latest training model")
        model_s3_prefix = get_latest_model_s3_prefix(S3_BUCKET, S3_PREFIX_BUILD_ARTIFACTS)

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=model_s3_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            if s3_key.endswith("/"):
                continue

            relative_path = s3_key.replace(model_s3_prefix, "").lstrip("/")
            local_file = local_model_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

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

def find_checkpoint_file(model_dir):

    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None
    
    patterns = ["**/*.ckpt",]
    
    for pattern in patterns:
        matches = list(model_dir.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None    