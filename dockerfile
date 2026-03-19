FROM python:3.11-slim
WORKDIR /TBClassifier
COPY . /TBClassifier
RUN python -m pip install --upgrade pip \
    && pip install --quiet pytorch-lightning \
                       optuna "optuna-integration[pytorch-lightning]" \
                       mlflow torchinfo plotly boto3
CMD ["python", "entrypoint.py"]