# Main Terraform configuration file for browser-based video game diffusion model infrastructure
# Version: 1.0.0
# Provider Version: AWS ~> 4.0

terraform {
  # Terraform version constraint
  required_version = ">= 1.0.0"

  # Required providers with version constraints
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }

  # Remote state configuration with encryption and locking
  backend "s3" {
    bucket         = "${var.project_name}-${var.environment}-tfstate"
    key            = "terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = "${var.project_name}-${var.environment}-tflock"
    kms_key_id     = var.kms_key_arn
    versioning     = true
  }
}

# Common resource tags for consistent labeling and management
locals {
  common_tags = {
    Project             = var.project_name
    Environment         = var.environment
    ManagedBy          = "terraform"
    CostCenter         = var.cost_center
    SecurityLevel      = var.security_level
    DataClassification = var.data_classification
    CreatedBy          = "terraform"
    CreatedDate        = timestamp()
  }

  # Resource naming convention
  resource_prefix = "${var.project_name}-${var.environment}"
}

# AWS provider configuration
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

# S3 bucket for static website hosting
module "static_website" {
  source = "./s3"

  bucket_name = "${local.resource_prefix}-static-website"
  versioning  = var.s3_versioning

  website_config = {
    index_document = "index.html"
    error_document = "error.html"
  }

  cors_rules = [
    {
      allowed_headers = ["*"]
      allowed_methods = ["GET"]
      allowed_origins = ["https://${var.static_website_domain}"]
      expose_headers  = ["ETag"]
      max_age_seconds = 3600
    }
  ]

  tags = local.common_tags
}

# CloudFront distribution for content delivery
module "cdn" {
  source = "./cloudfront"
  count  = var.enable_cdn ? 1 : 0

  distribution_name = "${local.resource_prefix}-cdn"
  domain_name      = var.static_website_domain
  price_class      = var.cdn_price_class

  origin_config = {
    domain_name = module.static_website.website_endpoint
    origin_id   = "S3-${local.resource_prefix}-static-website"
  }

  custom_error_responses = [
    {
      error_code            = 403
      response_code        = 200
      response_page_path   = "/index.html"
      error_caching_min_ttl = 0
    },
    {
      error_code            = 404
      response_code        = 200
      response_page_path   = "/index.html"
      error_caching_min_ttl = 0
    }
  ]

  cache_behavior = {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${local.resource_prefix}-static-website"

    forwarded_values = {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  tags = local.common_tags
}

# DynamoDB table for Terraform state locking
resource "aws_dynamodb_table" "terraform_lock" {
  name           = "${local.resource_prefix}-tflock"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"
  stream_enabled = false

  attribute {
    name = "LockID"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = local.common_tags
}

# KMS key for encrypting Terraform state
resource "aws_kms_key" "terraform_state" {
  description             = "KMS key for Terraform state encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

# Outputs for reference
output "static_website_endpoint" {
  description = "S3 static website endpoint"
  value       = module.static_website.website_endpoint
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = var.enable_cdn ? module.cdn[0].distribution_id : null
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = var.enable_cdn ? module.cdn[0].domain_name : null
}