/**
 * @fileoverview Training configuration for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs'; // v4.x
import { ModelConfig, ModelType, MemoryConstraints, BrowserFeature } from '../types/model';
import { TensorSpec, TensorFormat } from '../types/tensor';

/**
 * Memory configuration for training process
 */
export interface MemoryConfig {
    maxTensorAllocation: string;
    disposalStrategy: 'aggressive' | 'conservative';
    memoryGuardRatio: number;
    tensorPoolSize: number;
}

/**
 * Device memory information interface
 */
export interface DeviceMemoryInfo {
    totalGPUMemory: number;
    availableGPUMemory: number;
    totalRAM: number;
    browserMemoryLimit: number;
}

/**
 * Comprehensive training configuration interface
 */
export interface TrainingConfig {
    readonly learningRate: number;
    readonly batchSize: number;
    readonly epochs: number;
    readonly validationSplit: number;
    readonly optimizerType: string;
    readonly gradientClipping: number;
    readonly memoryConfig: MemoryConfig;
}

/**
 * Default training configuration with conservative memory settings
 */
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
    learningRate: 0.001,
    batchSize: 4,
    epochs: 10,
    validationSplit: 0.2,
    optimizerType: 'adam',
    gradientClipping: 1.0,
    memoryConfig: {
        maxTensorAllocation: '3GB',
        disposalStrategy: 'aggressive',
        memoryGuardRatio: 0.8,
        tensorPoolSize: 1000
    }
};

/**
 * Training constraints to ensure browser compatibility and performance
 */
export const TRAINING_CONSTRAINTS = {
    MIN_BATCH_SIZE: 1,
    MAX_BATCH_SIZE: 16,
    MIN_LEARNING_RATE: 0.0001,
    MAX_LEARNING_RATE: 0.01,
    MAX_EPOCHS: 100,
    MIN_MEMORY_PER_BATCH: 512 * 1024 * 1024, // 512MB
    MAX_TENSOR_SIZE: 2 * 1024 * 1024 * 1024,  // 2GB
    MIN_GPU_MEMORY: 1024 * 1024 * 1024,       // 1GB
    BROWSER_VERSIONS: {
        chrome: 90,
        firefox: 88,
        safari: 14,
        edge: 90
    }
};

/**
 * Parses memory string to bytes
 * @param memoryString Memory string (e.g., "2GB", "512MB")
 * @returns number of bytes
 */
function parseMemoryString(memoryString: string): number {
    const units = {
        KB: 1024,
        MB: 1024 * 1024,
        GB: 1024 * 1024 * 1024
    };
    const match = memoryString.match(/^(\d+)(KB|MB|GB)$/);
    if (!match) throw new Error(`Invalid memory string format: ${memoryString}`);
    return parseInt(match[1]) * units[match[2] as keyof typeof units];
}

/**
 * Validates training configuration against system constraints
 * @param config Training configuration to validate
 * @returns boolean indicating if configuration is valid
 */
export function validateTrainingConfig(config: TrainingConfig): boolean {
    // Check batch size constraints
    if (config.batchSize < TRAINING_CONSTRAINTS.MIN_BATCH_SIZE || 
        config.batchSize > TRAINING_CONSTRAINTS.MAX_BATCH_SIZE) {
        return false;
    }

    // Validate learning rate bounds
    if (config.learningRate < TRAINING_CONSTRAINTS.MIN_LEARNING_RATE || 
        config.learningRate > TRAINING_CONSTRAINTS.MAX_LEARNING_RATE) {
        return false;
    }

    // Check epoch count
    if (config.epochs <= 0 || config.epochs > TRAINING_CONSTRAINTS.MAX_EPOCHS) {
        return false;
    }

    // Validate validation split
    if (config.validationSplit < 0 || config.validationSplit >= 1) {
        return false;
    }

    // Check memory configuration
    const maxTensorBytes = parseMemoryString(config.memoryConfig.maxTensorAllocation);
    if (maxTensorBytes > TRAINING_CONSTRAINTS.MAX_TENSOR_SIZE) {
        return false;
    }

    // Validate memory guard ratio
    if (config.memoryConfig.memoryGuardRatio <= 0 || 
        config.memoryConfig.memoryGuardRatio > 1) {
        return false;
    }

    return true;
}

/**
 * Generates optimal training configuration based on device capabilities
 * @param modelConfig Model configuration
 * @param memoryInfo Device memory information
 * @returns Optimized training configuration
 */
export function getTrainingConfig(
    modelConfig: ModelConfig,
    memoryInfo: DeviceMemoryInfo
): TrainingConfig {
    // Start with default configuration
    let config: TrainingConfig = { ...DEFAULT_TRAINING_CONFIG };

    // Adjust batch size based on available memory
    const memoryPerBatch = TRAINING_CONSTRAINTS.MIN_MEMORY_PER_BATCH;
    const maxBatchSize = Math.floor(memoryInfo.availableGPUMemory / memoryPerBatch);
    config.batchSize = Math.min(
        maxBatchSize,
        TRAINING_CONSTRAINTS.MAX_BATCH_SIZE,
        modelConfig.memoryConstraints.tensorBufferSize / memoryPerBatch
    );

    // Scale learning rate with batch size
    config.learningRate = DEFAULT_TRAINING_CONFIG.learningRate * 
        (config.batchSize / DEFAULT_TRAINING_CONFIG.batchSize);

    // Configure memory management
    config.memoryConfig = {
        maxTensorAllocation: `${Math.floor(memoryInfo.availableGPUMemory * 0.8 / (1024 * 1024 * 1024))}GB`,
        disposalStrategy: memoryInfo.availableGPUMemory < TRAINING_CONSTRAINTS.MIN_GPU_MEMORY ? 
            'aggressive' : 'conservative',
        memoryGuardRatio: 0.8,
        tensorPoolSize: Math.floor(memoryInfo.availableGPUMemory / (2 * 1024 * 1024)) // 2MB per tensor
    };

    // Validate final configuration
    if (!validateTrainingConfig(config)) {
        throw new Error('Could not generate valid training configuration for device capabilities');
    }

    return config;
}

/**
 * Sets up training environment with memory management
 * @param config Training configuration
 */
export function setupTrainingEnvironment(config: TrainingConfig): void {
    // Configure TensorFlow.js memory management
    tf.engine().configureDebugMode(true);
    
    // Set up aggressive tensor disposal if needed
    if (config.memoryConfig.disposalStrategy === 'aggressive') {
        tf.engine().startScope(); // Enable automatic tensor cleanup
    }

    // Configure memory growth limits
    tf.engine().configureVirtualMachineOptions({
        maxMemoryMB: parseMemoryString(config.memoryConfig.maxTensorAllocation) / (1024 * 1024)
    });
}

/**
 * Cleans up training environment and releases resources
 */
export function cleanupTrainingEnvironment(): void {
    tf.engine().endScope();
    tf.engine().disposeVariables();
    tf.engine().purgeUnusedTensors();
}