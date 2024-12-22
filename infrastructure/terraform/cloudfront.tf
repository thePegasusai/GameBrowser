# AWS CloudFront Distribution Configuration for Browser-Based Video Game Diffusion Model
# Version: AWS Provider ~> 4.0

# Fetch existing ACM certificate for HTTPS
data "aws_acm_certificate" "static_website_cert" {
  domain      = var.static_website_domain
  statuses    = ["ISSUED"]
  most_recent = true
  provider    = aws.us-east-1  # ACM certificates must be in us-east-1 for CloudFront
}

# Create CloudFront Origin Access Identity for S3 bucket access
resource "aws_cloudfront_origin_access_identity" "static_website_oai" {
  comment = "${var.project_name}-${var.environment}-cf-oai"
}

# Configure CloudFront distribution
resource "aws_cloudfront_distribution" "static_website" {
  enabled             = var.enable_cdn
  price_class         = var.cdn_price_class
  aliases             = [var.static_website_domain]
  default_root_object = "index.html"
  is_ipv6_enabled     = true
  http_version        = "http2"

  # Origin configuration for S3 bucket
  origin {
    domain_name = aws_s3_bucket.static_website.bucket_domain_name
    origin_id   = "S3-${aws_s3_bucket.static_website.id}"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static_website_oai.cloudfront_access_identity_path
    }
  }

  # Default cache behavior settings
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static_website.id}"

    forwarded_values {
      query_string = false
      headers      = ["Origin", "Access-Control-Request-Headers", "Access-Control-Request-Method"]

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600  # 1 hour
    max_ttl                = 86400 # 24 hours
    compress               = true
  }

  # Custom error response for SPA routing
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  # SSL/TLS configuration
  viewer_certificate {
    acm_certificate_arn      = data.aws_acm_certificate.static_website_cert.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  # Geographic restrictions
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # Resource tags
  tags = {
    Name        = "${var.project_name}-${var.environment}-cf"
    Environment = var.environment
  }
}

# Output CloudFront distribution details
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution for cache invalidation"
  value       = aws_cloudfront_distribution.static_website.id
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.static_website.domain_name
}