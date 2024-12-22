#!/usr/bin/env bash

# Browser-based Video Game Diffusion Model Deployment Script
# Version: 1.0.0
# Dependencies:
# - aws-cli ^2.0.0
# - terraform ~> 1.0.0
# - jq ^1.6

set -euo pipefail
IFS=$'\n\t'

# Script constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"
readonly LOG_DIR="/var/log/bvgdm"
readonly DEPLOYMENT_STATE_DIR="/var/lib/bvgdm/deployments"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOCKFILE="/var/run/bvgdm_deploy.lock"

# Initialize logging with proper formatting and levels
initialize_logging() {
    local log_level="${1:-INFO}"
    
    # Create log directory if not exists
    mkdir -p "${LOG_DIR}"
    
    # Initialize log file with rotation
    readonly LOG_FILE="${LOG_DIR}/deploy_${TIMESTAMP}.log"
    touch "${LOG_FILE}"
    
    # Log format function
    log() {
        local level="$1"
        local message="$2"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] ${message}" | tee -a "${LOG_FILE}"
    }
    
    export -f log
    return 0
}

# Comprehensive validation of all deployment prerequisites
check_prerequisites() {
    log "INFO" "Checking deployment prerequisites..."
    
    # Check AWS CLI version and credentials
    if ! aws --version >/dev/null 2>&1; then
        log "ERROR" "AWS CLI not installed or not in PATH"
        return 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log "ERROR" "Invalid AWS credentials"
        return 1
    }
    
    # Check Terraform installation
    if ! terraform version >/dev/null 2>&1; then
        log "ERROR" "Terraform not installed or not in PATH"
        return 1
    }
    
    # Validate required environment variables
    local required_vars=(
        "AWS_ACCESS_KEY_ID"
        "AWS_SECRET_ACCESS_KEY"
        "AWS_REGION"
        "TF_VAR_environment"
        "DEPLOY_VERSION"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "ERROR" "Required environment variable ${var} is not set"
            return 1
        fi
    done
    
    return 0
}

# Deploy and validate AWS infrastructure with rollback capability
deploy_infrastructure() {
    local environment="$1"
    local version="$2"
    
    log "INFO" "Deploying infrastructure for environment: ${environment}, version: ${version}"
    
    # Create deployment state directory
    local deploy_state_dir="${DEPLOYMENT_STATE_DIR}/${version}"
    mkdir -p "${deploy_state_dir}"
    
    # Initialize Terraform
    cd "${TERRAFORM_DIR}"
    if ! terraform init -backend=true; then
        log "ERROR" "Failed to initialize Terraform"
        return 1
    fi
    
    # Select workspace
    if ! terraform workspace select "${environment}" 2>/dev/null; then
        terraform workspace new "${environment}"
    fi
    
    # Create and save deployment plan
    if ! terraform plan -out="${deploy_state_dir}/tfplan"; then
        log "ERROR" "Failed to create Terraform plan"
        return 1
    }
    
    # Apply infrastructure changes with timeout
    if ! timeout "${HEALTH_CHECK_TIMEOUT}" terraform apply -auto-approve "${deploy_state_dir}/tfplan"; then
        log "ERROR" "Failed to apply Terraform changes"
        rollback "${version}" "infrastructure"
        return 1
    }
    
    return 0
}

# Deploy and validate static content with versioning and rollback
deploy_static_content() {
    local bucket_name="$1"
    local distribution_id="$2"
    local version="$3"
    
    log "INFO" "Deploying static content version: ${version}"
    
    # Create versioned deployment directory
    local deploy_dir="${DEPLOYMENT_STATE_DIR}/${version}/static"
    mkdir -p "${deploy_dir}"
    
    # Build web application
    if ! npm run build; then
        log "ERROR" "Failed to build web application"
        return 1
    }
    
    # Backup current version
    aws s3 sync "s3://${bucket_name}" "${deploy_dir}/backup" --delete
    
    # Upload new version
    if ! aws s3 sync "./dist" "s3://${bucket_name}" \
        --delete \
        --cache-control "max-age=31536000" \
        --metadata "{\"version\":\"${version}\"}" \
        --exclude "*.html" \
        --exclude "*.json"; then
        log "ERROR" "Failed to upload static content"
        rollback "${version}" "static"
        return 1
    }
    
    # Upload HTML and JSON files with no-cache
    if ! aws s3 sync "./dist" "s3://${bucket_name}" \
        --delete \
        --cache-control "no-cache" \
        --metadata "{\"version\":\"${version}\"}" \
        --include "*.html" \
        --include "*.json"; then
        log "ERROR" "Failed to upload non-cached content"
        rollback "${version}" "static"
        return 1
    }
    
    # Create CloudFront invalidation
    if ! aws cloudfront create-invalidation \
        --distribution-id "${distribution_id}" \
        --paths "/*"; then
        log "ERROR" "Failed to invalidate CloudFront cache"
        return 1
    }
    
    return 0
}

# Handle automated rollback of failed deployments
rollback() {
    local version="$1"
    local component="$2"
    
    log "WARN" "Initiating rollback for ${component} version: ${version}"
    
    case "${component}" in
        "infrastructure")
            cd "${TERRAFORM_DIR}"
            terraform workspace select "${TF_VAR_environment}"
            terraform plan -destroy -out=destroy.tfplan
            terraform apply -auto-approve destroy.tfplan
            ;;
        "static")
            local deploy_dir="${DEPLOYMENT_STATE_DIR}/${version}/static"
            aws s3 sync "${deploy_dir}/backup" "s3://${bucket_name}" --delete
            aws cloudfront create-invalidation \
                --distribution-id "${distribution_id}" \
                --paths "/*"
            ;;
        *)
            log "ERROR" "Unknown component for rollback: ${component}"
            return 1
            ;;
    esac
    
    log "INFO" "Rollback completed successfully"
    return 0
}

# Comprehensive post-deployment cleanup and verification
cleanup() {
    local version="$1"
    
    log "INFO" "Performing deployment cleanup for version: ${version}"
    
    # Archive deployment logs
    local archive_dir="${LOG_DIR}/archive"
    mkdir -p "${archive_dir}"
    tar -czf "${archive_dir}/deploy_${version}_${TIMESTAMP}.tar.gz" "${LOG_FILE}"
    
    # Remove temporary files
    rm -f "${LOCKFILE}"
    rm -rf "${DEPLOYMENT_STATE_DIR}/${version}/static/backup"
    
    # Clean old deployment states (keep last 5)
    cd "${DEPLOYMENT_STATE_DIR}"
    ls -t | tail -n +6 | xargs -r rm -rf
    
    return 0
}

# Main deployment function
main() {
    # Acquire lock
    if ! mkdir "${LOCKFILE}" 2>/dev/null; then
        echo "Deployment already in progress"
        exit 1
    fi
    
    trap 'rm -rf "${LOCKFILE}"' EXIT
    
    # Initialize logging
    initialize_logging "${LOG_LEVEL}"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    }
    
    # Deploy infrastructure
    if ! deploy_infrastructure "${TF_VAR_environment}" "${DEPLOY_VERSION}"; then
        log "ERROR" "Infrastructure deployment failed"
        exit 1
    }
    
    # Get S3 bucket and CloudFront distribution from Terraform outputs
    local bucket_name=$(terraform output -raw static_website_endpoint)
    local distribution_id=$(terraform output -raw cloudfront_distribution_id)
    
    # Deploy static content
    if ! deploy_static_content "${bucket_name}" "${distribution_id}" "${DEPLOY_VERSION}"; then
        log "ERROR" "Static content deployment failed"
        exit 1
    }
    
    # Cleanup
    cleanup "${DEPLOY_VERSION}"
    
    log "INFO" "Deployment completed successfully"
    exit 0
}

# Execute main function
main "$@"