# Configure S3 bucket for static website content and model weights
# Version: AWS Provider ~> 4.0

# Main S3 bucket for static website content
resource "aws_s3_bucket" "static_website" {
  bucket = "${var.project_name}-${var.environment}-static"

  # Force destroy allowed in non-prod environments
  force_destroy = var.environment != "prod"

  tags = merge(var.tags, {
    Name        = "${var.project_name}-${var.environment}-static"
    Environment = var.environment
    Purpose     = "Static content hosting for video game diffusion model"
  })
}

# Enable versioning for content management and rollback capability
resource "aws_s3_bucket_versioning" "static_website_versioning" {
  bucket = aws_s3_bucket.static_website.id
  versioning_configuration {
    status = var.s3_versioning ? "Enabled" : "Disabled"
  }
}

# Block all public access for security
resource "aws_s3_bucket_public_access_block" "static_website_public_access" {
  bucket = aws_s3_bucket.static_website.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Configure CORS for browser access
resource "aws_s3_bucket_cors_configuration" "static_website_cors" {
  bucket = aws_s3_bucket.static_website.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["https://${var.static_website_domain}"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3600
  }
}

# Enable server-side encryption for data at rest
resource "aws_s3_bucket_server_side_encryption_configuration" "static_website_encryption" {
  bucket = aws_s3_bucket.static_website.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Configure lifecycle rules for version management
resource "aws_s3_bucket_lifecycle_configuration" "static_website_lifecycle" {
  bucket = aws_s3_bucket.static_website.id

  rule {
    id     = "version-cleanup"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Bucket policy allowing CloudFront access
resource "aws_s3_bucket_policy" "static_website_policy" {
  bucket = aws_s3_bucket.static_website.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowCloudFrontOAIAccess"
        Effect    = "Allow"
        Principal = {
          AWS = aws_cloudfront_origin_access_identity.static_website_oai.iam_arn
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.static_website.arn}/*"
      }
    ]
  })
}

# Output bucket details for other modules
output "static_website_bucket" {
  description = "S3 bucket details for static website content"
  value = {
    id                          = aws_s3_bucket.static_website.id
    arn                         = aws_s3_bucket.static_website.arn
    bucket_domain_name          = aws_s3_bucket.static_website.bucket_domain_name
    bucket_regional_domain_name = aws_s3_bucket.static_website.bucket_regional_domain_name
  }
}