terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ── Variables ────────────────────────────────────────────────────────────────

variable "region" {
  default = "ap-south-1"
}

variable "bucket_name" {
  default = "my-bucket-20260618"
}

variable "sagemaker_role_name" {
  default = "sagemaker-training-role-test"
}

# ── S3 Bucket ────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "ml_lake" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_versioning" "ml_lake" {
  bucket = aws_s3_bucket.ml_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_lake" {
  bucket = aws_s3_bucket.ml_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_lake" {
  bucket                  = aws_s3_bucket.ml_lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM Role ─────────────────────────────────────────────────────────────────

resource "aws_iam_role" "sagemaker" {
  name = var.sagemaker_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# ── ECR Repository ────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "ml_inference" {
  name                 = "propensity-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "bucket_name" {
  value = aws_s3_bucket.ml_lake.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.ml_inference.repository_url
}
