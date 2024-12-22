# AWS Certificate Manager Configuration for Browser-Based Video Game Diffusion Model
# Version: AWS Provider ~> 4.0

# Create ACM certificate for the static website domain
resource "aws_acm_certificate" "static_website" {
  # Certificate must be in us-east-1 for CloudFront compatibility
  provider = aws.us-east-1

  domain_name       = var.static_website_domain
  validation_method = "DNS"

  tags = {
    Name                = "${var.project_name}-${var.environment}-cert"
    Environment         = var.environment
    AutoRenew          = "true"
    SecurityCompliance = "required"
    ManagedBy          = "terraform"
  }

  # Create new certificate before destroying the old one
  lifecycle {
    create_before_destroy = true
  }
}

# Create DNS records for certificate validation
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.static_website.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id         = data.aws_route53_zone.static_website.zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}

# Validate ACM certificate using DNS validation
resource "aws_acm_certificate_validation" "static_website" {
  provider                = aws.us-east-1
  certificate_arn         = aws_acm_certificate.static_website.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]

  timeouts {
    create = "45m"
  }
}

# Output certificate ARN for use in CloudFront distribution
output "acm_certificate_arn" {
  description = "ARN of the ACM certificate for HTTPS configuration"
  value       = aws_acm_certificate.static_website.arn
}

# Output certificate validation details for monitoring
output "certificate_validation_details" {
  description = "Certificate validation record details for DNS verification"
  value = {
    certificate_arn = aws_acm_certificate.static_website.arn
    domain_name     = var.static_website_domain
    status         = aws_acm_certificate_validation.static_website.id != "" ? "VALIDATED" : "PENDING"
  }
  sensitive = false
}

# Data source for retrieving the Route53 zone
data "aws_route53_zone" "static_website" {
  name         = var.static_website_domain
  private_zone = false
}