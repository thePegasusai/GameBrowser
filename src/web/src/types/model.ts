/**
 * @fileoverview Core type definitions for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { Tensor, DataType } from '@tensorflow/tfjs-core'; // v4.x
import { TensorSpec } from './tensor';

/**
 * Supported model architectures
 */
export enum ModelType {
    DiT = 'DiT',
    VAE = 'VAE'
}

/**
 * Model processing states with memory tracking
 */
export enum ModelState {
    LOADING = 'LOADING',
    READY = 'READY',
    TRAINING = 'TRAINING',
    GENERATING = 'GENERATING',
    ERROR = 'ERROR',
    OUT_OF_MEMORY = 'OUT_OF_MEMORY'
}

/**
 * Required browser features for model operation
 */
export enum BrowserFeature {
    WEBGL2 = 'WEBGL2',
    WEBWORKER = 'WEBWORKER',
    INDEXEDDB = 'INDEXEDDB',
    SHAREDARRAYBUFFER = 'SHAREDARRAYBUFFER'
}

/**
 * DiT model configuration interface
 */
export interface DiTConfig {
    readonly numLayers: number;
    readonly hiddenSize: number;
    readonly numHeads: number;
    readonly intermediateSize: number;
    readonly dropoutRate: number;
    readonly inputSpec: TensorSpec;
    readonly outputSpec: TensorSpec;
}

/**
 * VAE model configuration interface
 */
export interface VAEConfig {
    readonly encoderLayers: number[];
    readonly decoderLayers: number[];
    readonly latentDim: number;
    readonly inputSpec: TensorSpec;
    readonly latentSpec: TensorSpec;
}

/**
 * Memory constraints configuration
 */
export interface MemoryConstraints {
    readonly maxRAMUsage: number;  // in bytes
    readonly maxGPUMemory: number; // in bytes
    readonly tensorBufferSize: number;
    readonly enableMemoryTracking: boolean;
}

/**
 * WebGL capabilities interface
 */
export interface WebGLCapabilities {
    readonly version: number;
    readonly maxTextureSize: number;
    readonly maxRenderBufferSize: number;
    readonly vendor: string;
    readonly renderer: string;
}

/**
 * Browser capabilities and validation
 */
export interface BrowserCapabilities {
    readonly requiredFeatures: BrowserFeature[];
    readonly webgl: WebGLCapabilities;
    readonly isCompatible: boolean;
}

/**
 * Memory usage metrics
 */
export interface MemoryUsage {
    readonly heapUsed: number;
    readonly heapTotal: number;
    readonly gpuUsed: number;
    readonly tensorCount: number;
    readonly bufferCount: number;
}

/**
 * Browser performance metrics
 */
export interface BrowserMetrics {
    readonly fps: number;
    readonly renderTime: number;
    readonly gpuUtilization: number;
    readonly powerEfficiency: number;
}

/**
 * Model performance metrics
 */
export interface ModelMetrics {
    readonly inferenceTime: number;
    readonly memoryMetrics: MemoryUsage;
    readonly gpuUtilization: number;
    readonly browserMetrics: BrowserMetrics;
}

/**
 * Memory validation result type
 */
export type MemoryStatus = {
    available: number;
    required: number;
    canAllocate: boolean;
};

/**
 * Browser compatibility result type
 */
export type BrowserStatus = {
    supported: boolean;
    missingFeatures: BrowserFeature[];
    warnings: string[];
};

/**
 * Runtime validation type
 */
export type ModelValidation = {
    validate(): Promise<boolean>;
    checkMemory(): Promise<MemoryStatus>;
    validateBrowser(): Promise<BrowserStatus>;
};

/**
 * Enhanced model configuration interface with validation
 */
export interface ModelConfig {
    readonly modelType: ModelType;
    readonly ditConfig?: DiTConfig;
    readonly vaeConfig?: VAEConfig;
    readonly memoryConstraints: MemoryConstraints;
    readonly browserCapabilities: BrowserCapabilities;
    readonly validation: ModelValidation;
}

/**
 * Type guard for ModelConfig
 */
export function isModelConfig(value: unknown): value is ModelConfig {
    return (
        typeof value === 'object' &&
        value !== null &&
        'modelType' in value &&
        'memoryConstraints' in value &&
        'browserCapabilities' in value &&
        'validation' in value
    );
}

/**
 * Type guard for MemoryConstraints
 */
export function isMemoryConstraints(value: unknown): value is MemoryConstraints {
    return (
        typeof value === 'object' &&
        value !== null &&
        'maxRAMUsage' in value &&
        'maxGPUMemory' in value &&
        'tensorBufferSize' in value &&
        'enableMemoryTracking' in value
    );
}

/**
 * Type guard for BrowserCapabilities
 */
export function isBrowserCapabilities(value: unknown): value is BrowserCapabilities {
    return (
        typeof value === 'object' &&
        value !== null &&
        'requiredFeatures' in value &&
        'webgl' in value &&
        'isCompatible' in value &&
        Array.isArray((value as BrowserCapabilities).requiredFeatures)
    );
}

/**
 * Default memory constraints
 */
export const DEFAULT_MEMORY_CONSTRAINTS: MemoryConstraints = {
    maxRAMUsage: 4 * 1024 * 1024 * 1024, // 4GB
    maxGPUMemory: 2 * 1024 * 1024 * 1024, // 2GB
    tensorBufferSize: 256 * 1024 * 1024,   // 256MB
    enableMemoryTracking: true
};

/**
 * Default browser feature requirements
 */
export const DEFAULT_BROWSER_FEATURES: BrowserFeature[] = [
    BrowserFeature.WEBGL2,
    BrowserFeature.WEBWORKER,
    BrowserFeature.INDEXEDDB
];