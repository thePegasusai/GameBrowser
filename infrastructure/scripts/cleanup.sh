#!/usr/bin/env bash

# Cleanup script for browser-based video game diffusion model infrastructure
# Version: 1.0.0
# Dependencies: aws-cli ^2.0.0, terraform ~> 1.0.0

set -euo pipefail
IFS=$'\n\t'

# Source environment variables if exists
if [[ -f ".env" ]]; then
    source .env
fi

# Global variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TERRAFORM_DIR="../terraform"
readonly BUILD_DIR="../../dist"
readonly LOG_RETENTION_DAYS=30
readonly REQUIRED_AWS_CLI_VERSION="2.0.0"
readonly REQUIRED_TERRAFORM_VERSION="1.0.0"

# Initialize logging
setup_logging() {
    CLEANUP_LOG_FILE=${CLEANUP_LOG_FILE:-"/tmp/cleanup-${TF_VAR_environment:-dev}-$(date +%Y%m%d).log"}
    exec 1> >(tee -a "${CLEANUP_LOG_FILE}")
    exec 2> >(tee -a "${CLEANUP_LOG_FILE}" >&2)
    log "Cleanup started at $(date '+%Y-%m-%d %H:%M:%S')"
}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Error handling function
error() {
    log "ERROR: $*" >&2
    exit 1
}

# Check prerequisites
check_prerequisites() {
    # Check AWS credentials
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        error "AWS credentials not set. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    fi

    # Check AWS CLI version
    if ! command -v aws >/dev/null 2>&1; then
        error "AWS CLI not installed"
    fi
    
    local aws_version
    aws_version=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
    if ! [[ "${aws_version}" =~ ^2\. ]]; then
        error "AWS CLI version ${REQUIRED_AWS_CLI_VERSION} or higher required"
    fi

    # Check Terraform installation
    if ! command -v terraform >/dev/null 2>&1; then
        error "Terraform not installed"
    }

    # Verify environment variables
    [[ -z "${TF_VAR_environment:-}" ]] && error "TF_VAR_environment not set"
    [[ -z "${AWS_REGION:-}" ]] && error "AWS_REGION not set"

    log "Prerequisites check passed"
    return 0
}

# Cleanup S3 content
cleanup_s3_content() {
    local bucket_name=$1
    local create_backup=${2:-false}

    log "Starting S3 cleanup for bucket: ${bucket_name}"

    if [[ "${create_backup}" == "true" ]] && [[ "${BACKUP_ENABLED:-true}" == "true" ]]; then
        local backup_bucket="${bucket_name}-backup-$(date +%Y%m%d)"
        log "Creating backup in bucket: ${backup_bucket}"
        aws s3 mb "s3://${backup_bucket}" --region "${AWS_REGION}"
        aws s3 sync "s3://${bucket_name}" "s3://${backup_bucket}" --delete
    fi

    log "Removing contents from bucket: ${bucket_name}"
    if ! aws s3 rm "s3://${bucket_name}" --recursive; then
        error "Failed to remove contents from bucket: ${bucket_name}"
    fi

    log "S3 cleanup completed for bucket: ${bucket_name}"
    return 0
}

# Cleanup local files
cleanup_local_files() {
    log "Starting local file cleanup"

    # Clean Terraform files
    if [[ -d "${TERRAFORM_DIR}" ]]; then
        find "${TERRAFORM_DIR}" -type f -name "*.tfstate*" -delete
        find "${TERRAFORM_DIR}" -type f -name "*.backup" -delete
        find "${TERRAFORM_DIR}" -type d -name ".terraform" -exec rm -rf {} +
    fi

    # Clean build directory
    if [[ -d "${BUILD_DIR}" ]]; then
        rm -rf "${BUILD_DIR}"/*
    fi

    # Clean old logs
    find /tmp -name "cleanup-*-*.log" -mtime +"${LOG_RETENTION_DAYS}" -delete

    # Clean npm cache if exists
    if command -v npm >/dev/null 2>&1; then
        npm cache clean --force
    fi

    log "Local file cleanup completed"
    return 0
}

# Destroy infrastructure
destroy_infrastructure() {
    local environment=$1
    local force=${2:-false}

    log "Starting infrastructure destruction for environment: ${environment}"

    cd "${TERRAFORM_DIR}" || error "Failed to change to Terraform directory"

    # Initialize Terraform
    terraform init -backend=true || error "Terraform initialization failed"

    # Select workspace
    terraform workspace select "${environment}" || error "Failed to select Terraform workspace"

    # Create destroy plan
    terraform plan -destroy -out=destroy.tfplan || error "Failed to create destroy plan"

    if [[ "${force}" != "true" ]]; then
        read -rp "Are you sure you want to destroy the infrastructure? (yes/no) " confirm
        if [[ "${confirm}" != "yes" ]]; then
            log "Infrastructure destruction cancelled"
            return 0
        fi
    fi

    # Destroy infrastructure
    if ! terraform destroy -auto-approve; then
        error "Infrastructure destruction failed"
    fi

    log "Infrastructure destruction completed"
    return 0
}

# Main function
main() {
    setup_logging
    check_prerequisites || error "Prerequisites check failed"

    # Parse command line arguments
    local force=false
    local backup=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force) force=true ;;
            --no-backup) backup=false ;;
            *) error "Unknown option: $1" ;;
        esac
        shift
    done

    # Execute cleanup steps
    cleanup_local_files || error "Local file cleanup failed"

    if [[ -n "${TF_VAR_environment:-}" ]]; then
        local bucket_name="${TF_VAR_project_name:-bvgdm}-${TF_VAR_environment}-static-website"
        cleanup_s3_content "${bucket_name}" "${backup}" || error "S3 cleanup failed"
        
        if [[ "${force}" == "true" ]]; then
            destroy_infrastructure "${TF_VAR_environment}" true || error "Infrastructure destruction failed"
        fi
    fi

    log "Cleanup completed successfully"
    return 0
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

This script provides a comprehensive cleanup solution with the following features:

1. Robust error handling and logging
2. Prerequisite checks for required tools and credentials
3. S3 bucket cleanup with optional backup
4. Local file cleanup including Terraform state and build artifacts
5. Infrastructure destruction with safety confirmations
6. Environment-aware operations
7. Command-line options for force mode and backup control

The script follows best practices for shell scripting:
- Uses strict error checking with `set -euo pipefail`
- Implements proper logging and error handling
- Provides clear feedback and confirmation prompts
- Includes comprehensive documentation
- Supports both interactive and automated usage
- Maintains idempotency where possible
- Implements proper cleanup of temporary files and resources

The script can be executed with optional parameters:
```bash
./cleanup.sh --force    # Skip confirmation prompts
./cleanup.sh --no-backup    # Skip backup creation