# Minimal Terraform stub for deploying FinSight as a container on AWS.
# This is a starting point: it provisions an ECR repository for the image and
# a CloudWatch log group. Extend with ECS Fargate / App Runner as needed.
#
#   terraform init && terraform apply

terraform {
  required_version = ">= 1.5"
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

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "app_name" {
  type    = string
  default = "finsight"
}

resource "aws_ecr_repository" "finsight" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "finsight" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 14
}

output "ecr_repository_url" {
  value = aws_ecr_repository.finsight.repository_url
}
