# AWS Route53 Configuration for Browser-Based Video Game Diffusion Model
# Version: AWS Provider ~> 4.0

# Fetch existing Route53 hosted zone for the domain
data "aws_route53_zone" "static_website" {
  name         = var.static_website_domain
  private_zone = false
}

# Create A record for the static website pointing to CloudFront
resource "aws_route53_record" "static_website_a" {
  zone_id = data.aws_route53_zone.static_website.zone_id
  name    = var.static_website_domain
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.static_website.domain_name
    zone_id               = aws_cloudfront_distribution.static_website.hosted_zone_id
    evaluate_target_health = false
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-dns"
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "static-website-dns"
  }
}

# Create HTTPS health check for the static website
resource "aws_route53_health_check" "static_website" {
  fqdn              = var.static_website_domain
  port              = 443
  type              = "HTTPS"
  resource_path     = "/"
  failure_threshold = "3"
  request_interval  = "30"

  tags = {
    Name        = "${var.project_name}-${var.environment}-health-check"
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "static-website-health"
  }
}

# Output Route53 zone ID for ACM certificate validation
output "route53_zone_id" {
  description = "Route53 hosted zone ID for the static website domain"
  value       = data.aws_route53_zone.static_website.zone_id
}

# Output fully qualified domain name for the static website
output "website_dns_name" {
  description = "Fully qualified domain name for the static website"
  value       = aws_route53_record.static_website_a.fqdn
}