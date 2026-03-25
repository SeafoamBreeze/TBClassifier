import os
import boto3
from config import S3_BUCKET, DATASET_PATH, OPTUNA_DIR
from pathlib import Path

def upload_file_to_s3(file_name, s3_bucket, s3_destination):
    s3 = boto3.client("s3")
    try:    
        print(f"upload_file_to_s3(): Uploading to {upload_file_to_s3}")
        s3.upload_file(file_name, s3_bucket, s3_destination)
    except Exception as e:
        print(f"upload_file_to_s3(): Upload failed: {e}")

def download_latest_optuna_study():
    download_from_s3(bucket=S3_BUCKET, prefix="tuning-artifact/latest/optuna_studies", local_dir=str(OPTUNA_DIR))
    verify_item_count("tuning-artifact/latest/optuna_studies", {".db"})

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

def verify_item_count(dataset_path, ext_dict):

    if not dataset_path.exists():
        print(f"Path not found: {dataset_path}")
        return
    
    print(f"Looking in {dataset_path}")
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            count = sum(
                1 for f in class_dir.rglob("*")
                if f.suffix.lower() in ext_dict
            )
            print(f"{class_dir.name}: {count}")