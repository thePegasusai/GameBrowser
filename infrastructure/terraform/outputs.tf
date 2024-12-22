# S3 bucket outputs
output "s3_bucket_id" {
  description = "ID of the S3 bucket hosting static website content"
  value       = aws_s3_bucket.static_website.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket hosting static website content"
  value       = aws_s3_bucket.static_website.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket for static website hosting"
  value       = aws_s3_bucket.static_website.bucket_domain_name
}

# CloudFront outputs
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution for content delivery"
  value       = aws_cloudfront_distribution.static_website.id
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.static_website.domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "Route 53 zone ID for the CloudFront distribution"
  value       = aws_cloudfront_distribution.static_website.hosted_zone_id
}

# Combined outputs for external reference
output "static_website_infrastructure" {
  description = "Combined infrastructure details for the static website"
  value = {
    s3_bucket = {
      id           = aws_s3_bucket.static_website.id
      arn          = aws_s3_bucket.static_website.arn
      domain_name  = aws_s3_bucket.static_website.bucket_domain_name
    }
    cloudfront = {
      distribution_id = aws_cloudfront_distribution.static_website.id
      domain_name    = aws_cloudfront_distribution.static_website.domain_name
      hosted_zone_id = aws_cloudfront_distribution.static_website.hosted_zone_id
    }
    environment     = var.environment
    project_name    = var.project_name
  }
}

# Deployment validation outputs
output "deployment_ready" {
  description = "Indicates if the infrastructure is ready for deployment"
  value = {
    cdn_enabled     = var.enable_cdn
    s3_versioning   = var.s3_versioning
    price_class     = var.cdn_price_class
    website_domain  = var.static_website_domain
  }
}