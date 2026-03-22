import os
import subprocess

RUN_TUNING = os.environ.get("RUN_TUNING", "false").lower() == "true"
RUN_TRAINING = os.environ.get("RUN_TRAINING", "false").lower() == "true"

if __name__ == "__main__":
    print(f"RUN_TUNING = {RUN_TUNING}")
    print(f"RUN_TRAINING = {RUN_TRAINING}")
    if RUN_TUNING:
        print("Running TBClassifier_tuning.py...")
        subprocess.run(["python", "TBClassifier_tuning.py"], check=True)
    else:
        print("Tuning Skipped")

    if RUN_TRAINING:
        print("Running TBClassifier_training.py...")
        subprocess.run(["python", "TBClassifier_training.py"], check=True)
    else:
        print("Training skipped")