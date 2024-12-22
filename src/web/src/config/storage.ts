/**
 * @fileoverview Storage configuration for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { Tensor } from '@tensorflow/tfjs-core'; // v4.x
import { ModelWeights } from '../types/model';

/**
 * Interface defining enhanced storage configuration options
 */
export interface StorageConfig {
    readonly name: string;
    readonly version: number;
    readonly stores: Record<string, string>;
    readonly validation: ValidationConfig;
}

/**
 * Interface defining cache configuration with memory tracking
 */
export interface CacheConfig {
    readonly name: string;
    readonly maxAge: number;
    readonly maxItems: number;
    readonly maxSize: number;
    readonly memoryTracking: MemoryTracking;
}

/**
 * Interface for storage validation configuration
 */
export interface ValidationConfig {
    readonly maxKeyLength: number;
    readonly maxStoreSize: number;
    readonly requiredIndices: string[];
}

/**
 * Interface for memory tracking configuration
 */
export interface MemoryTracking {
    readonly interval: number;
    readonly warningThreshold: number;
    readonly cleanupThreshold: number;
}

/**
 * Enhanced IndexedDB configuration with validation
 */
export const INDEXED_DB_CONFIG: StorageConfig = {
    name: 'bvgdm-store',
    version: 1,
    stores: {
        MODEL_WEIGHTS: 'modelWeights',
        CHECKPOINTS: 'checkpoints',
        TRAINING_DATA: 'trainingData'
    },
    validation: {
        maxKeyLength: 100,
        maxStoreSize: 1024 * 1024 * 1024, // 1GB
        requiredIndices: ['id', 'timestamp']
    }
} as const;

/**
 * Cache API configuration with memory tracking
 */
export const CACHE_CONFIG: CacheConfig = {
    name: 'bvgdm-cache',
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    maxItems: 1000,
    maxSize: 4 * 1024 * 1024 * 1024, // 4GB
    memoryTracking: {
        interval: 5000, // 5 seconds
        warningThreshold: 0.8, // 80%
        cleanupThreshold: 0.9 // 90%
    }
} as const;

/**
 * Storage quota configuration and monitoring
 */
export const STORAGE_QUOTAS = {
    MIN_AVAILABLE: 100 * 1024 * 1024, // 100MB
    MAX_MODEL_SIZE: 1024 * 1024 * 1024, // 1GB
    MAX_CHECKPOINT_SIZE: 512 * 1024 * 1024, // 512MB
    MONITORING: {
        CHECK_INTERVAL: 10000, // 10 seconds
        WARNING_THRESHOLD: 0.85, // 85%
        ERROR_THRESHOLD: 0.95 // 95%
    }
} as const;

/**
 * Storage validation utilities
 */
export class StorageValidation {
    /**
     * Validates available storage quota against requirements
     * @returns Promise resolving to validation result
     */
    static async validateQuota(): Promise<boolean> {
        try {
            if ('storage' in navigator && 'estimate' in navigator.storage) {
                const estimate = await navigator.storage.estimate();
                const available = estimate.quota! - estimate.usage!;
                return available >= STORAGE_QUOTAS.MIN_AVAILABLE;
            }
            return false;
        } catch (error) {
            console.error('Storage quota validation failed:', error);
            return false;
        }
    }

    /**
     * Checks availability of required storage mechanisms
     * @returns Promise resolving to storage availability status
     */
    static async checkStorageAvailability(): Promise<{
        indexedDB: boolean;
        cacheAPI: boolean;
        quota: boolean;
    }> {
        const results = {
            indexedDB: false,
            cacheAPI: false,
            quota: false
        };

        try {
            // Check IndexedDB
            results.indexedDB = 'indexedDB' in window;

            // Check Cache API
            results.cacheAPI = 'caches' in window;

            // Check Storage Quota
            results.quota = await this.validateQuota();

            return results;
        } catch (error) {
            console.error('Storage availability check failed:', error);
            return results;
        }
    }

    /**
     * Validates model weights for storage
     * @param weights Model weights to validate
     * @returns boolean indicating if weights are valid for storage
     */
    static validateModelWeights(weights: ModelWeights): boolean {
        if (!weights || !weights.id || !weights.weights) {
            return false;
        }

        // Check total size
        const totalSize = weights.weights.reduce((size, tensor) => {
            return size + (tensor.size * Float32Array.BYTES_PER_ELEMENT);
        }, 0);

        return totalSize <= STORAGE_QUOTAS.MAX_MODEL_SIZE;
    }

    /**
     * Monitors storage usage and triggers cleanup if needed
     * @returns Promise resolving to current usage percentage
     */
    static async monitorStorageUsage(): Promise<number> {
        try {
            const estimate = await navigator.storage.estimate();
            const usagePercent = (estimate.usage! / estimate.quota!) * 100;

            if (usagePercent >= STORAGE_QUOTAS.MONITORING.ERROR_THRESHOLD) {
                throw new Error('Storage quota exceeded');
            }

            if (usagePercent >= STORAGE_QUOTAS.MONITORING.WARNING_THRESHOLD) {
                console.warn('Storage usage warning:', usagePercent.toFixed(2) + '%');
            }

            return usagePercent;
        } catch (error) {
            console.error('Storage monitoring failed:', error);
            throw error;
        }
    }
}

/**
 * Type guard for StorageConfig
 */
export function isStorageConfig(value: unknown): value is StorageConfig {
    return (
        typeof value === 'object' &&
        value !== null &&
        'name' in value &&
        'version' in value &&
        'stores' in value &&
        'validation' in value
    );
}

/**
 * Type guard for CacheConfig
 */
export function isCacheConfig(value: unknown): value is CacheConfig {
    return (
        typeof value === 'object' &&
        value !== null &&
        'name' in value &&
        'maxAge' in value &&
        'maxItems' in value &&
        'maxSize' in value &&
        'memoryTracking' in value
    );
}