# Configure Terraform and required providers with strict version constraints
# and security-focused settings for the browser-based video game diffusion model
terraform {
  # Require Terraform 1.0 or higher for stability and security features
  required_version = ">= 1.0"

  # Configure required providers with specific versions and security checksums
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
      # Provider checksums for security verification
      checksums = {
        darwin_amd64 = "sha256:e33c91f1652b2d0a9d52b1f4dddb3c8dca43b9c9678934f1bc6d4962a5916b44"
        linux_amd64  = "sha256:a4ff8b5c94b6c8a282a0f7167f23a1c02d68deaaa46717ce3f2465df86a151b5"
      }
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
      # Provider checksums for security verification
      checksums = {
        darwin_amd64 = "sha256:a8c8c9f9c2f99a6e650b5660eb49a8c4b0f534d1824c1ded8d8876839e9e6843"
        linux_amd64  = "sha256:f98e6d01e4a11e11e4bb9c11dcf0a1d5c1f3f1c42f87e537c39e6f3c5d7d2d3a"
      }
    }
  }

  # Configure local backend with workspace support for state management
  backend "local" {
    workspace_dir = "../terraform.tfstate.d"
  }

  # Enable experimental features for enhanced functionality
  experiments = ["module_variable_optional_attrs"]
}

# Configure AWS provider with enhanced security and monitoring settings
provider "aws" {
  # Use validated region from variables
  region = var.aws_region

  # Configure default tags for resource tracking and management
  default_tags {
    tags = {
      Project       = "bvgdm"
      ManagedBy     = "terraform"
      Environment   = terraform.workspace
      Application   = "browser-video-game-diffusion"
      SecurityLevel = "high"
      CostCenter    = "ml-research"
    }
  }

  # Configure retry behavior for improved reliability
  retry_mode   = "standard"
  max_retries  = 3

  # Configure IAM role assumption for enhanced security
  assume_role {
    role_arn     = var.aws_role_arn
    session_name = "terraform-bvgdm"
  }

  # Configure service endpoints for regional access
  endpoints {
    s3         = "s3.${var.aws_region}.amazonaws.com"
    cloudfront = "cloudfront.amazonaws.com"
  }

  # Configure tag filtering for resource management
  ignore_tags {
    keys = ["temporary", "test"]
  }
}

# Configure random provider for secure resource naming
provider "random" {
  # Random provider inherits default configuration
}