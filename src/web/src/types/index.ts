/**
 * @fileoverview Main type definitions index file for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { Tensor } from '@tensorflow/tfjs-core'; // v4.x

// Import model-related types
import {
    ModelConfig,
    DiTConfig,
    VAEConfig,
    ModelType,
    ModelState,
    BrowserFeature,
    MemoryConstraints,
    WebGLCapabilities,
    BrowserCapabilities,
    MemoryUsage,
    BrowserMetrics,
    ModelMetrics,
    MemoryStatus,
    BrowserStatus,
    ModelValidation,
    isModelConfig,
    isMemoryConstraints,
    isBrowserCapabilities,
    DEFAULT_MEMORY_CONSTRAINTS,
    DEFAULT_BROWSER_FEATURES
} from './model';

// Import tensor-related types
import {
    TensorSpec,
    TensorDimensions,
    TensorFormat,
    TensorMemoryInfo,
    DefaultTensorSpec,
    isTensorMemoryInfo,
    isTensorSpec
} from './tensor';

// Import video-related types
import {
    VideoFrame,
    VideoFormat,
    VideoMetadata,
    VideoProcessingConfig,
    VideoProcessingState,
    VideoProcessingError,
    isVideoFrame,
    isVideoMetadata,
    isVideoProcessingConfig
} from './video';

/**
 * Global application configuration interface
 * Combines model, video, and tensor configurations
 */
export interface AppConfig {
    readonly modelConfig: ModelConfig;
    readonly videoConfig: VideoProcessingConfig;
    readonly tensorSpec: TensorSpec;
    readonly memoryConstraints: MemoryConstraints;
}

/**
 * Interface for runtime browser compatibility validation
 */
export interface BrowserCompatibilityCheck {
    readonly webGLSupported: boolean;
    readonly webWorkersSupported: boolean;
    readonly indexedDBSupported: boolean;
    readonly browserVersion: string;
    readonly webGLVersion: string;
    readonly requiredFeatures: BrowserFeature[];
    validate(): Promise<BrowserStatus>;
}

/**
 * Interface for tracking and managing global memory usage
 */
export interface GlobalMemoryManager {
    readonly totalMemoryUsage: number;
    readonly tensorMemoryUsage: number;
    readonly videoMemoryUsage: number;
    readonly modelMemoryUsage: number;
    readonly isMemoryWarning: boolean;
    readonly memoryPool: 'webgl' | 'cpu';
    checkMemoryStatus(): Promise<MemoryStatus>;
    disposeUnusedTensors(): Promise<void>;
    getMemoryUsageSnapshot(): MemoryUsageSnapshot;
}

/**
 * Union type for application states
 */
export type AppState = ModelState | VideoProcessingState;

/**
 * Type for runtime type validation results
 */
export type TypeValidationResult = {
    readonly isValid: boolean;
    readonly errors: string[];
    readonly warnings: string[];
};

/**
 * Type for memory usage snapshots
 */
export type MemoryUsageSnapshot = {
    readonly timestamp: number;
    readonly usage: GlobalMemoryManager;
    readonly tensors: TensorMemoryInfo[];
};

// Re-export all types for centralized access
export {
    // Model types
    ModelConfig,
    DiTConfig,
    VAEConfig,
    ModelType,
    ModelState,
    BrowserFeature,
    MemoryConstraints,
    WebGLCapabilities,
    BrowserCapabilities,
    MemoryUsage,
    BrowserMetrics,
    ModelMetrics,
    MemoryStatus,
    BrowserStatus,
    ModelValidation,
    isModelConfig,
    isMemoryConstraints,
    isBrowserCapabilities,
    DEFAULT_MEMORY_CONSTRAINTS,
    DEFAULT_BROWSER_FEATURES,

    // Tensor types
    TensorSpec,
    TensorDimensions,
    TensorFormat,
    TensorMemoryInfo,
    DefaultTensorSpec,
    isTensorMemoryInfo,
    isTensorSpec,

    // Video types
    VideoFrame,
    VideoFormat,
    VideoMetadata,
    VideoProcessingConfig,
    VideoProcessingState,
    VideoProcessingError,
    isVideoFrame,
    isVideoMetadata,
    isVideoProcessingConfig
};