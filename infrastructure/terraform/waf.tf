# AWS WAF v2 Configuration for Browser-Based Video Game Diffusion Model
# Version: AWS Provider ~> 4.0

# Create WAF Web ACL for CloudFront
resource "aws_wafv2_web_acl" "cloudfront_waf" {
  name        = "${var.project_name}-${var.environment}-waf"
  description = "WAF rules for browser-based video game diffusion model with enhanced security and monitoring"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  # AWS Managed Core Rule Set
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesCommonRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }

  # AWS Known Bad Inputs Rule Set
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesKnownBadInputsRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }

  # IP-based Rate Limiting
  rule {
    name     = "IPRateLimit"
    priority = 3

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "IPRateLimitMetric"
      sampled_requests_enabled  = true
    }
  }

  # Global visibility config
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name               = "${var.project_name}-${var.environment}-waf-metrics"
    sampled_requests_enabled  = true
  }

  # Resource tags
  tags = {
    Name        = "${var.project_name}-${var.environment}-waf"
    Environment = var.environment
    Service     = "WAF"
    ManagedBy   = "Terraform"
  }
}

# Associate WAF Web ACL with CloudFront distribution
resource "aws_wafv2_web_acl_association" "cloudfront_waf_association" {
  resource_arn = aws_cloudfront_distribution.static_website.arn
  web_acl_arn  = aws_wafv2_web_acl.cloudfront_waf.arn
}

# Output WAF Web ACL details
output "waf_web_acl_id" {
  description = "ID of the WAF Web ACL"
  value       = aws_wafv2_web_acl.cloudfront_waf.id
}

output "waf_web_acl_arn" {
  description = "ARN of the WAF Web ACL"
  value       = aws_wafv2_web_acl.cloudfront_waf.arn
}