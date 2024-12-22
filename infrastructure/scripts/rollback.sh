#!/usr/bin/env bash

# Browser-based Video Game Diffusion Model Rollback Script
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
readonly BACKUP_DIR="/var/lib/bvgdm/backups"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOCKFILE="/var/run/bvgdm_rollback.lock"

# Source common functions from deploy.sh
source "${SCRIPT_DIR}/deploy.sh"

# Initialize logging
initialize_logging() {
    local log_file="${LOG_DIR}/rollback_${TIMESTAMP}.log"
    mkdir -p "${LOG_DIR}"
    
    # Log format function
    log() {
        local level="$1"
        local message="$2"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] ${message}" | tee -a "${log_file}"
    }
    
    export -f log
    export LOG_FILE="${log_file}"
}

# Validate rollback version compatibility
check_rollback_version() {
    local version="$1"
    local environment="$2"
    
    log "INFO" "Validating rollback version: ${version} for environment: ${environment}"
    
    # Check version format
    if ! [[ ${version} =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        log "ERROR" "Invalid version format: ${version}"
        return 1
    }
    
    # Verify S3 backup exists
    local backup_path="s3://${environment}-backups/${version}"
    if ! aws s3 ls "${backup_path}" >/dev/null 2>&1; then
        log "ERROR" "Backup not found: ${backup_path}"
        return 1
    }
    
    # Check Terraform state backup
    local tf_backup="${BACKUP_DIR}/${version}/terraform.tfstate"
    if [[ ! -f "${tf_backup}" ]]; then
        log "ERROR" "Terraform state backup not found: ${tf_backup}"
        return 1
    }
    
    # Validate browser compatibility metadata
    local metadata_file="${BACKUP_DIR}/${version}/metadata.json"
    if [[ -f "${metadata_file}" ]]; then
        local required_webgl_version=$(jq -r '.requirements.webgl_version' "${metadata_file}")
        local required_indexeddb_version=$(jq -r '.requirements.indexeddb_version' "${metadata_file}")
        
        # Verify compatibility
        if [[ "${required_webgl_version}" > "2.0" ]] || [[ "${required_indexeddb_version}" > "3.0" ]]; then
            log "ERROR" "Version ${version} has incompatible browser requirements"
            return 1
        fi
    fi
    
    return 0
}

# Roll back infrastructure state
rollback_infrastructure() {
    local version="$1"
    local environment="$2"
    
    log "INFO" "Rolling back infrastructure to version: ${version}"
    
    # Create backup of current state
    local current_backup="${BACKUP_DIR}/pre_rollback_${TIMESTAMP}"
    mkdir -p "${current_backup}"
    
    if ! terraform state pull > "${current_backup}/terraform.tfstate"; then
        log "ERROR" "Failed to backup current Terraform state"
        return 1
    }
    
    # Initialize Terraform
    cd "${TERRAFORM_DIR}"
    if ! terraform init -backend=true; then
        log "ERROR" "Failed to initialize Terraform"
        return 1
    }
    
    # Select workspace
    if ! terraform workspace select "${environment}"; then
        log "ERROR" "Failed to select Terraform workspace: ${environment}"
        return 1
    }
    
    # Restore previous state
    local tf_backup="${BACKUP_DIR}/${version}/terraform.tfstate"
    if ! terraform state push "${tf_backup}"; then
        log "ERROR" "Failed to restore Terraform state"
        terraform state push "${current_backup}/terraform.tfstate"
        return 1
    }
    
    # Apply previous infrastructure state
    if ! terraform apply -auto-approve; then
        log "ERROR" "Failed to apply previous infrastructure state"
        terraform state push "${current_backup}/terraform.tfstate"
        return 1
    }
    
    return 0
}

# Roll back static content
rollback_static_content() {
    local bucket_name="$1"
    local distribution_id="$2"
    local version="$3"
    
    log "INFO" "Rolling back static content to version: ${version}"
    
    # Backup current content
    local current_backup="${BACKUP_DIR}/pre_rollback_${TIMESTAMP}/static"
    mkdir -p "${current_backup}"
    
    if ! aws s3 sync "s3://${bucket_name}" "${current_backup}" --delete; then
        log "ERROR" "Failed to backup current static content"
        return 1
    }
    
    # Restore previous version
    local backup_path="s3://${bucket_name}-backups/${version}"
    if ! aws s3 sync "${backup_path}" "s3://${bucket_name}" \
        --delete \
        --metadata "{\"rollback_version\":\"${version}\",\"rollback_timestamp\":\"${TIMESTAMP}\"}" \
        --cache-control "no-cache"; then
        log "ERROR" "Failed to restore static content"
        aws s3 sync "${current_backup}" "s3://${bucket_name}" --delete
        return 1
    }
    
    # Invalidate CloudFront cache
    if ! aws cloudfront create-invalidation \
        --distribution-id "${distribution_id}" \
        --paths "/*" \
        --query 'Invalidation.Id' \
        --output text; then
        log "ERROR" "Failed to invalidate CloudFront cache"
        return 1
    }
    
    return 0
}

# Perform post-rollback cleanup
cleanup_rollback() {
    log "INFO" "Performing post-rollback cleanup"
    
    # Archive rollback logs
    local archive_dir="${LOG_DIR}/archive"
    mkdir -p "${archive_dir}"
    tar -czf "${archive_dir}/rollback_${TIMESTAMP}.tar.gz" "${LOG_FILE}"
    
    # Clean up old backups beyond retention period
    find "${BACKUP_DIR}" -type d -mtime +${BACKUP_RETENTION_DAYS} -exec rm -rf {} +
    
    # Remove lock file
    rm -f "${LOCKFILE}"
    
    # Generate rollback report
    local report_file="${LOG_DIR}/rollback_report_${TIMESTAMP}.json"
    jq -n \
        --arg timestamp "${TIMESTAMP}" \
        --arg version "${ROLLBACK_VERSION}" \
        --arg status "completed" \
        '{timestamp: $timestamp, version: $version, status: $status}' > "${report_file}"
    
    return 0
}

# Main rollback function
main() {
    # Acquire lock
    if ! mkdir "${LOCKFILE}" 2>/dev/null; then
        echo "Rollback already in progress"
        exit 1
    fi
    
    trap 'rm -rf "${LOCKFILE}"' EXIT
    
    # Initialize logging
    initialize_logging
    
    # Validate environment variables
    if [[ -z "${ROLLBACK_VERSION:-}" ]] || [[ -z "${TF_VAR_environment:-}" ]]; then
        log "ERROR" "Required environment variables not set"
        exit 1
    }
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    }
    
    # Validate rollback version
    if ! check_rollback_version "${ROLLBACK_VERSION}" "${TF_VAR_environment}"; then
        log "ERROR" "Rollback version validation failed"
        exit 1
    }
    
    # Roll back infrastructure
    if ! rollback_infrastructure "${ROLLBACK_VERSION}" "${TF_VAR_environment}"; then
        log "ERROR" "Infrastructure rollback failed"
        exit 1
    }
    
    # Get S3 bucket and CloudFront distribution from Terraform outputs
    local bucket_name=$(terraform output -raw static_website_endpoint)
    local distribution_id=$(terraform output -raw cloudfront_distribution_id)
    
    # Roll back static content
    if ! rollback_static_content "${bucket_name}" "${distribution_id}" "${ROLLBACK_VERSION}"; then
        log "ERROR" "Static content rollback failed"
        exit 1
    }
    
    # Cleanup
    cleanup_rollback
    
    log "INFO" "Rollback completed successfully"
    exit 0
}

# Execute main function
main "$@"