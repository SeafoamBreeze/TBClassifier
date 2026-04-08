# Function to upload to AwS lambda

import json
import os
import boto3
import urllib3


def get_fargate_public_ip():
    print("--- [1] Starting Fargate IP lookup ---")
    region = "ap-southeast-1"
    cluster = "advanced-frog-ig3pzu"
    service = "tbclassifier-inference"

    ecs = boto3.client("ecs", region_name=region)

    print("Querying ECS for running tasks...")
    tasks = ecs.list_tasks(cluster=cluster, serviceName=service)
    print(f"Tasks found: {tasks['taskArns']}")

    if not tasks["taskArns"]:
        raise Exception("No running tasks found for the service!")
    task_arn = tasks["taskArns"][0]

    print(f"Describing task details for: {task_arn}")
    task_details = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
    attachments = task_details["tasks"][0]["attachments"][0]["details"]

    eni_id = next(
        d["value"] for d in attachments if d["name"] == "networkInterfaceId"
    )
    print(f"Found ENI ID: {eni_id}")

    print("Describing Network Interface to get Public IP...")
    ec2 = boto3.client("ec2", region_name=region)
    eni_details = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])

    public_ip = eni_details["NetworkInterfaces"][0]["Association"]["PublicIp"]
    print(f"Success! Live Fargate IP is: {public_ip}")
    return public_ip


def lambda_handler(event, context):
    print("=== NEW INVOCATION RECEIVED ===")
    print(f"Incoming event keys: {list(event.keys())}")

    try:
        # 1. Get current Fargate IP
        fargate_ip = get_fargate_public_ip()
        fargate_url = f"http://{fargate_ip}:8000/predict"

        # 2. Extract payload sent by Web App
        print("Extracting payload from event...")
        if "body" in event and event["body"]:
            print("Payload found in event['body']. Decoding JSON...")
            payload = json.loads(event["body"])
        else:
            print("No body key found. Using raw event as payload.")
            payload = event

        print(f"Payload keys to send: {list(payload.keys())}")

        # 3. Proxy the call to Fargate
        print(f"Forwarding request to Fargate at: {fargate_url}")
        http = urllib3.PoolManager()
        encoded_payload = json.dumps(payload).encode("utf-8")

        print("Sending POST request to Fargate...")
        response = http.request(
            "POST",
            fargate_url,
            body=encoded_payload,
            headers={"Content-Type": "application/json"},
            timeout=25.0,
        )

        print(f"Fargate responded with HTTP Status: {response.status}")
        raw_body = response.data.decode("utf-8")
        print(f"Response: {raw_body}")
        
        # 4. Strict format return
        print("Constructing final response for AWS...")
        final_response = {
            "statusCode": int(response.status),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": raw_body,
        }
        print("Successfully built response object. Returning to AWS.")
        return final_response

    except Exception as e:
        print(f"⚠️ ERROR CAUGHT: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }