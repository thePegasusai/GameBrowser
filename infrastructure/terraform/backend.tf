# Backend configuration for Terraform state management
# Version: terraform ~> 1.0
# Purpose: Manages secure state storage and locking for infrastructure management

terraform {
  # Configure S3 backend with workspace support and encryption
  backend "s3" {
    bucket               = "${var.project_name}-${var.environment}-tfstate"
    key                  = "terraform.tfstate"
    region              = "${var.aws_region}"
    encrypt             = true
    dynamodb_table      = "${var.project_name}-${var.environment}-tflock"
    workspace_key_prefix = "workspaces"
  }
}

# S3 bucket for storing Terraform state with versioning and encryption
resource "aws_s3_bucket" "terraform_state" {
  bucket = "${var.project_name}-${var.environment}-tfstate"

  # Enable versioning to maintain state history
  versioning {
    enabled = true
  }

  # Configure server-side encryption
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-tfstate"
    Environment = var.environment
    Purpose     = "Terraform State Storage"
  }
}

# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "${var.project_name}-${var.environment}-tflock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  # Configure table attributes
  attribute {
    name = "LockID"
    type = "S"
  }

  # Enable point-in-time recovery for data protection
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-tflock"
    Environment = var.environment
    Purpose     = "Terraform State Locking"
  }
}