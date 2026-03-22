import os
import boto3

def download_dataset(bucket, prefix, local_dir):

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
                print(f"Downloading {s3_key} → {local_path}")
                s3.download_file(bucket, s3_key, local_path)
            except Exception as e:
                print(f"Download failed: {e}")


