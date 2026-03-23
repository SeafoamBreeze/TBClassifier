import os
import boto3
from config import BUCKET, DATASET_PATH, STUDY_DB, TRACKING_DB

def upload_file_to_s3(file_name, s3_bucket, s3_destination):
    s3 = boto3.client("s3")
    try:    
        print(f"upload_file_to_s3(): Uploading to {upload_file_to_s3}")
        s3.upload_file(file_name, s3_bucket, s3_destination)
    except Exception as e:
        print(f"upload_file_to_s3(): Upload failed: {e}")


def download_dataset_from_s3():

    download_from_s3(BUCKET, str(DATASET_PATH/"train"), str(DATASET_PATH/"train"))
    download_from_s3(BUCKET, str(DATASET_PATH/"test"), str(DATASET_PATH/"test"))

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


