import os
import subprocess
from utils.s3_utils import download_dataset_from_s3

RUN_01_TUNING_SCRIPT = os.environ.get("RUN_01_TUNING_SCRIPT", "false").lower() == "true"
RUN_02_TRAINING_SCRIPT = os.environ.get("RUN_02_TRAINING_SCRIPT", "false").lower() == "true"

if __name__ == "__main__":

    download_dataset_from_s3()

    print(f"RUN_01_TUNING_SCRIPT = {RUN_01_TUNING_SCRIPT}")
    print(f"RUN_02_TRAINING_SCRIPT = {RUN_02_TRAINING_SCRIPT}")

    if RUN_01_TUNING_SCRIPT:
        print("Running TBClassifier_tuning.py...")
        subprocess.run(["python", "01_TBClassifier_tuning.py"], check=True)
    else:
        print("Tuning Skipped")

    if RUN_02_TRAINING_SCRIPT:
        print("Running TBClassifier_training.py...")
        subprocess.run(["python", "02_TBClassifier_training.py"], check=True)
    else:
        print("Training skipped")