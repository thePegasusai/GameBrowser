# AWS region configuration with validation for supported regions
variable "aws_region" {
  description = "AWS region for resource deployment with validation for supported regions"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^(us|eu|ap|sa|ca|me|af)-[a-z]+-\\d+$", var.aws_region))
    error_message = "Must be a valid AWS region identifier"
  }
}

# Project name configuration with length and character constraints
variable "project_name" {
  description = "Name of the project for resource naming with length and character constraints"
  type        = string
  default     = "bvgdm"

  validation {
    condition     = can(regex("^[a-z0-9-]{3,16}$", var.project_name))
    error_message = "Project name must be 3-16 characters, lowercase alphanumeric with hyphens"
  }
}

# Environment name for resource tagging and configuration
variable "environment" {
  description = "Environment name for resource tagging and configuration"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

# S3 bucket versioning configuration
variable "s3_versioning" {
  description = "Enable versioning for S3 buckets storing model weights and assets"
  type        = bool
  default     = true
}

# Static website domain configuration with DNS validation
variable "static_website_domain" {
  description = "Domain name for the static website with DNS validation"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\\.[a-z]{2,}$", var.static_website_domain))
    error_message = "Must be a valid domain name"
  }
}

# CloudFront CDN enablement configuration
variable "enable_cdn" {
  description = "Flag to enable/disable CloudFront CDN for content delivery"
  type        = bool
  default     = true
}

# CloudFront price class configuration with validation
variable "cdn_price_class" {
  description = "Price class for CloudFront distribution with validation"
  type        = string
  default     = "PriceClass_100"

  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.cdn_price_class)
    error_message = "Must be a valid CloudFront price class"
  }
}

# Common resource tags
variable "tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default = {
    Project    = "BVGDM"
    ManagedBy  = "Terraform"
  }
}