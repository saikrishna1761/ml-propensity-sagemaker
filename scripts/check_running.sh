#!/bin/bash

REGIONS=("ap-south-1" "us-east-1" "us-west-2" "eu-west-1")

echo "=============================="
echo " LOCAL DOCKER CONTAINERS"
echo "=============================="
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=============================="
echo " AWS SAGEMAKER ENDPOINTS"
echo "=============================="
for region in "${REGIONS[@]}"; do
  result=$(aws sagemaker list-endpoints --region "$region" \
    --query "Endpoints[*].{Name:EndpointName, Status:EndpointStatus}" \
    --output text 2>/dev/null)
  if [ -n "$result" ]; then
    echo "[$region] $result"
  else
    echo "[$region] none"
  fi
done

echo ""
echo "=============================="
echo " AWS SAGEMAKER TRAINING JOBS (InProgress)"
echo "=============================="
for region in "${REGIONS[@]}"; do
  result=$(aws sagemaker list-training-jobs --region "$region" \
    --status-equals InProgress \
    --query "TrainingJobSummaries[*].TrainingJobName" \
    --output text 2>/dev/null)
  if [ -n "$result" ]; then
    echo "[$region] $result"
  else
    echo "[$region] none"
  fi
done

echo ""
echo "=============================="
echo " AWS EC2 INSTANCES (running)"
echo "=============================="
for region in "${REGIONS[@]}"; do
  result=$(aws ec2 describe-instances --region "$region" \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[*].{ID:InstanceId,Type:InstanceType}" \
    --output text 2>/dev/null)
  if [ -n "$result" ]; then
    echo "[$region] $result"
  else
    echo "[$region] none"
  fi
done

echo ""
echo "=============================="
echo " AWS RDS INSTANCES (available)"
echo "=============================="
for region in "${REGIONS[@]}"; do
  result=$(aws rds describe-db-instances --region "$region" \
    --query "DBInstances[*].{ID:DBInstanceIdentifier,Status:DBInstanceStatus}" \
    --output text 2>/dev/null)
  if [ -n "$result" ]; then
    echo "[$region] $result"
  else
    echo "[$region] none"
  fi
done

echo ""
echo "Done."
