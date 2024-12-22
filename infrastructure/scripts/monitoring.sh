#!/bin/bash

# Browser-Based Video Game Diffusion Model - Infrastructure Monitoring Script
# Version: 1.0.0
# Dependencies: aws-cli (2.x), jq (1.6)

set -euo pipefail

# Load environment variables with defaults
AWS_PROFILE=${AWS_PROFILE:-"default"}
MONITORING_INTERVAL=${MONITORING_INTERVAL:-300}
ALERT_THRESHOLD=${ALERT_THRESHOLD:-90}
OUTPUT_DIR=${OUTPUT_DIR:-"/var/log/bvgdm/monitoring"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
CACHE_DURATION=${CACHE_DURATION:-3600}

# Performance thresholds from technical specifications
PERFORMANCE_THRESHOLDS='{
    "frame_generation_time": 50,
    "memory_usage": 4096,
    "gpu_utilization": 80,
    "model_load_time": 5000,
    "training_step_time": 200
}'

# Security thresholds for WAF monitoring
SECURITY_THRESHOLDS='{
    "max_4xx_rate": 5,
    "max_5xx_rate": 1,
    "max_blocked_requests": 1000,
    "rate_limit_threshold": 1800
}'

# Logging function with timestamp and level
log() {
    local level=$1
    local message=$2
    if [[ "${LOG_LEVELS[$LOG_LEVEL]:-0}" -le "${LOG_LEVELS[$level]:-0}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message"
    fi
}

# Initialize logging
declare -A LOG_LEVELS=([DEBUG]=0 [INFO]=1 [WARN]=2 [ERROR]=3)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check CloudFront metrics
check_cloudfront_metrics() {
    local distribution_id=$1
    local period=${2:-300}
    
    log "INFO" "Checking CloudFront metrics for distribution $distribution_id"
    
    # Fetch CloudFront metrics using AWS CLI
    metrics=$(aws cloudwatch get-metric-data \
        --metric-data-queries '[
            {
                "Id": "requests",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/CloudFront",
                        "MetricName": "Requests",
                        "Dimensions": [{"Name": "DistributionId", "Value": "'$distribution_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Sum"
                }
            },
            {
                "Id": "errors4xx",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/CloudFront",
                        "MetricName": "4xxErrorRate",
                        "Dimensions": [{"Name": "DistributionId", "Value": "'$distribution_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Average"
                }
            },
            {
                "Id": "errors5xx",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/CloudFront",
                        "MetricName": "5xxErrorRate",
                        "Dimensions": [{"Name": "DistributionId", "Value": "'$distribution_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Average"
                }
            }
        ]' \
        --start-time $(date -u -d "5 minutes ago" +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ))
    
    echo "$metrics" | jq '.'
}

# Function to check WAF metrics
check_waf_metrics() {
    local web_acl_id=$1
    local period=${2:-300}
    
    log "INFO" "Checking WAF metrics for Web ACL $web_acl_id"
    
    # Fetch WAF metrics using AWS CLI
    metrics=$(aws cloudwatch get-metric-data \
        --metric-data-queries '[
            {
                "Id": "blocked",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/WAFV2",
                        "MetricName": "BlockedRequests",
                        "Dimensions": [{"Name": "WebACL", "Value": "'$web_acl_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Sum"
                }
            },
            {
                "Id": "allowed",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/WAFV2",
                        "MetricName": "AllowedRequests",
                        "Dimensions": [{"Name": "WebACL", "Value": "'$web_acl_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Sum"
                }
            },
            {
                "Id": "rateBlocked",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/WAFV2",
                        "MetricName": "RateBasedRuleBlocks",
                        "Dimensions": [{"Name": "WebACL", "Value": "'$web_acl_id'"}]
                    },
                    "Period": '$period',
                    "Stat": "Sum"
                }
            }
        ]' \
        --start-time $(date -u -d "5 minutes ago" +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ))
    
    echo "$metrics" | jq '.'
}

# Function to monitor application performance metrics
monitor_performance_metrics() {
    local thresholds=$1
    local output_file="$OUTPUT_DIR/performance_metrics.json"
    
    log "INFO" "Monitoring application performance metrics"
    
    # Collect browser performance metrics via CloudWatch Custom Metrics
    metrics=$(aws cloudwatch get-metric-data \
        --metric-data-queries '[
            {
                "Id": "frameGenTime",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "BVGDM/Performance",
                        "MetricName": "FrameGenerationTime"
                    },
                    "Period": 300,
                    "Stat": "Average"
                }
            },
            {
                "Id": "memoryUsage",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "BVGDM/Performance",
                        "MetricName": "MemoryUsage"
                    },
                    "Period": 300,
                    "Stat": "Maximum"
                }
            },
            {
                "Id": "gpuUtilization",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "BVGDM/Performance",
                        "MetricName": "GPUUtilization"
                    },
                    "Period": 300,
                    "Stat": "Average"
                }
            }
        ]' \
        --start-time $(date -u -d "5 minutes ago" +%Y-%m-%dT%H:%M:%SZ) \
        --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ))
    
    echo "$metrics" | jq '.' > "$output_file"
}

# Function to generate comprehensive monitoring report
generate_report() {
    local output_format=$1
    local report_type=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="$OUTPUT_DIR/monitoring_report_${timestamp}.${output_format}"
    
    log "INFO" "Generating monitoring report in $output_format format"
    
    # Combine all metrics into a single report
    {
        echo "Browser-Based Video Game Diffusion Model - Monitoring Report"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo
        echo "=== CloudFront Metrics ==="
        check_cloudfront_metrics "$CLOUDFRONT_DISTRIBUTION_ID" "$MONITORING_INTERVAL"
        echo
        echo "=== WAF Metrics ==="
        check_waf_metrics "$WAF_WEB_ACL_ID" "$MONITORING_INTERVAL"
        echo
        echo "=== Performance Metrics ==="
        monitor_performance_metrics "$PERFORMANCE_THRESHOLDS"
    } > "$report_file"
    
    log "INFO" "Report generated: $report_file"
}

# Main monitoring loop
main() {
    log "INFO" "Starting monitoring script"
    
    # Validate environment
    if ! command -v aws >/dev/null 2>&1; then
        log "ERROR" "AWS CLI not found. Please install aws-cli v2"
        exit 1
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        log "ERROR" "jq not found. Please install jq"
        exit 1
    }
    
    # Main monitoring loop
    while true; do
        log "INFO" "Running monitoring checks"
        
        # Generate monitoring report
        generate_report "json" "full"
        
        # Check for alerts
        if [[ -f "$OUTPUT_DIR/alerts.json" ]]; then
            log "WARN" "Alerts detected, check $OUTPUT_DIR/alerts.json"
        fi
        
        # Wait for next interval
        sleep "$MONITORING_INTERVAL"
    done
}

# Start monitoring if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi