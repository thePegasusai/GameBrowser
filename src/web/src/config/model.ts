/**
 * @fileoverview Core model configuration and architecture settings for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs'; // v4.x
import { 
    ModelConfig, 
    ModelType,
    DiTConfig,
    VAEConfig,
    BrowserCapabilities,
    MemoryConstraints,
    BrowserFeature,
    ModelValidation,
    MemoryStatus,
    BrowserStatus,
    WebGLCapabilities
} from '../types/model';
import { TensorFormat, TensorSpec, DefaultTensorSpec } from '../types/tensor';

/**
 * Default model configuration with safety margins
 */
export const DEFAULT_MODEL_CONFIG = {
    modelType: ModelType.DiT,
    batchSize: 4,
    inputShape: [256, 256, 3],
    hiddenSize: 768,
    numLayers: 24,
    numHeads: 12,
    patchSize: 2,
    memoryLimit: 3072, // MB
    safetyMargin: 0.2
} as const;

/**
 * Memory allocation and limit configurations
 */
export const MODEL_MEMORY_LIMITS = {
    MIN_MEMORY: 2048,  // MB
    MAX_MEMORY: 4096,  // MB
    BUFFER_PERCENTAGE: 0.2,
    ALLOCATION_CHUNK: 256 // MB
} as const;

/**
 * Browser and WebGL requirements
 */
export const BROWSER_REQUIREMENTS = {
    MIN_WEBGL_VERSION: 2.0,
    MIN_TEXTURE_SIZE: 4096,
    REQUIRED_EXTENSIONS: ['EXT_float_blend', 'OES_texture_float'] as const
} as const;

/**
 * Generates optimal model configuration based on device capabilities
 * @param deviceInfo - Device capability information
 * @param browserCapabilities - Browser feature support and WebGL capabilities
 * @returns Optimized ModelConfig instance
 */
export function getModelConfig(
    deviceInfo: { memory: number; gpu: WebGLCapabilities },
    browserCapabilities: BrowserCapabilities
): ModelConfig {
    // Validate browser compatibility
    if (!browserCapabilities.isCompatible) {
        throw new Error('Browser does not meet minimum requirements');
    }

    // Calculate available memory with safety margin
    const availableMemory = Math.min(
        deviceInfo.memory * (1 - DEFAULT_MODEL_CONFIG.safetyMargin),
        MODEL_MEMORY_LIMITS.MAX_MEMORY
    );

    // Configure input/output tensor specifications
    const inputSpec = new DefaultTensorSpec(
        [DEFAULT_MODEL_CONFIG.batchSize, ...DEFAULT_MODEL_CONFIG.inputShape],
        'float32',
        TensorFormat.NHWC
    );

    const outputSpec = new DefaultTensorSpec(
        [DEFAULT_MODEL_CONFIG.batchSize, ...DEFAULT_MODEL_CONFIG.inputShape],
        'float32',
        TensorFormat.NHWC
    );

    // Configure DiT model
    const ditConfig: DiTConfig = {
        numLayers: DEFAULT_MODEL_CONFIG.numLayers,
        hiddenSize: DEFAULT_MODEL_CONFIG.hiddenSize,
        numHeads: DEFAULT_MODEL_CONFIG.numHeads,
        intermediateSize: DEFAULT_MODEL_CONFIG.hiddenSize * 4,
        dropoutRate: 0.1,
        inputSpec,
        outputSpec
    };

    // Configure VAE model
    const vaeConfig: VAEConfig = {
        encoderLayers: [64, 128, 256, 512],
        decoderLayers: [512, 256, 128, 64],
        latentDim: 256,
        inputSpec,
        latentSpec: new DefaultTensorSpec([DEFAULT_MODEL_CONFIG.batchSize, 256], 'float32', TensorFormat.NHWC)
    };

    // Configure memory constraints
    const memoryConstraints: MemoryConstraints = {
        maxRAMUsage: availableMemory * 1024 * 1024,
        maxGPUMemory: deviceInfo.gpu.maxTextureSize * deviceInfo.gpu.maxTextureSize * 4,
        tensorBufferSize: MODEL_MEMORY_LIMITS.ALLOCATION_CHUNK * 1024 * 1024,
        enableMemoryTracking: true
    };

    // Create validation handlers
    const validation: ModelValidation = {
        async validate(): Promise<boolean> {
            const memoryStatus = await this.checkMemory();
            const browserStatus = await this.validateBrowser();
            return memoryStatus.canAllocate && browserStatus.supported;
        },

        async checkMemory(): Promise<MemoryStatus> {
            const info = await tf.memory();
            return {
                available: memoryConstraints.maxRAMUsage,
                required: info.numBytes,
                canAllocate: info.numBytes < memoryConstraints.maxRAMUsage
            };
        },

        async validateBrowser(): Promise<BrowserStatus> {
            const missingFeatures = browserCapabilities.requiredFeatures.filter(
                feature => !isBrowserFeatureSupported(feature)
            );

            return {
                supported: missingFeatures.length === 0,
                missingFeatures,
                warnings: getWebGLWarnings(browserCapabilities.webgl)
            };
        }
    };

    return {
        modelType: DEFAULT_MODEL_CONFIG.modelType,
        ditConfig,
        vaeConfig,
        memoryConstraints,
        browserCapabilities,
        validation
    };
}

/**
 * Validates model configuration against system constraints
 * @param config - Model configuration to validate
 * @param capabilities - Browser capabilities
 * @returns Validation result with detailed error information
 */
export function validateModelConfig(
    config: ModelConfig,
    capabilities: BrowserCapabilities
): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate WebGL version
    if (capabilities.webgl.version < BROWSER_REQUIREMENTS.MIN_WEBGL_VERSION) {
        errors.push(`WebGL ${BROWSER_REQUIREMENTS.MIN_WEBGL_VERSION} is required`);
    }

    // Validate texture size limits
    if (capabilities.webgl.maxTextureSize < BROWSER_REQUIREMENTS.MIN_TEXTURE_SIZE) {
        errors.push(`Minimum texture size of ${BROWSER_REQUIREMENTS.MIN_TEXTURE_SIZE} is required`);
    }

    // Validate memory constraints
    if (config.memoryConstraints.maxRAMUsage > MODEL_MEMORY_LIMITS.MAX_MEMORY * 1024 * 1024) {
        errors.push(`Memory requirement exceeds maximum limit of ${MODEL_MEMORY_LIMITS.MAX_MEMORY}MB`);
    }

    // Validate model dimensions
    const totalParams = calculateModelParameters(config);
    if (totalParams * 4 > config.memoryConstraints.maxRAMUsage) {
        errors.push('Model size exceeds available memory');
    }

    return {
        isValid: errors.length === 0,
        errors
    };
}

/**
 * Dynamically adjusts model configuration based on runtime conditions
 * @param currentConfig - Current model configuration
 * @param metrics - Current performance metrics
 * @returns Adjusted model configuration
 */
export function adjustModelConfig(
    currentConfig: ModelConfig,
    metrics: { memory: number; inferenceTime: number }
): ModelConfig {
    const adjustedConfig = { ...currentConfig };

    // Adjust batch size based on memory usage
    if (metrics.memory > MODEL_MEMORY_LIMITS.MAX_MEMORY * 0.9) {
        adjustedConfig.ditConfig = {
            ...adjustedConfig.ditConfig!,
            inputSpec: new DefaultTensorSpec(
                [Math.max(1, DEFAULT_MODEL_CONFIG.batchSize - 1), ...DEFAULT_MODEL_CONFIG.inputShape],
                'float32',
                TensorFormat.NHWC
            )
        };
    }

    // Adjust model size based on inference time
    if (metrics.inferenceTime > 50) { // 50ms target
        adjustedConfig.ditConfig = {
            ...adjustedConfig.ditConfig!,
            numLayers: Math.max(12, adjustedConfig.ditConfig!.numLayers - 2)
        };
    }

    return adjustedConfig;
}

/**
 * Helper function to check browser feature support
 */
function isBrowserFeatureSupported(feature: BrowserFeature): boolean {
    switch (feature) {
        case BrowserFeature.WEBGL2:
            return tf.findBackend('webgl') !== undefined;
        case BrowserFeature.WEBWORKER:
            return typeof Worker !== 'undefined';
        case BrowserFeature.INDEXEDDB:
            return typeof indexedDB !== 'undefined';
        case BrowserFeature.SHAREDARRAYBUFFER:
            return typeof SharedArrayBuffer !== 'undefined';
        default:
            return false;
    }
}

/**
 * Helper function to get WebGL-specific warnings
 */
function getWebGLWarnings(capabilities: WebGLCapabilities): string[] {
    const warnings: string[] = [];
    
    if (capabilities.maxTextureSize < BROWSER_REQUIREMENTS.MIN_TEXTURE_SIZE) {
        warnings.push('Limited texture size may impact performance');
    }

    for (const ext of BROWSER_REQUIREMENTS.REQUIRED_EXTENSIONS) {
        if (!capabilities.renderer.includes(ext)) {
            warnings.push(`Missing WebGL extension: ${ext}`);
        }
    }

    return warnings;
}

/**
 * Helper function to calculate total model parameters
 */
function calculateModelParameters(config: ModelConfig): number {
    const { ditConfig } = config;
    if (!ditConfig) return 0;

    const embedDim = ditConfig.hiddenSize;
    const ffnDim = ditConfig.intermediateSize;
    const numLayers = ditConfig.numLayers;
    const numHeads = ditConfig.numHeads;

    return (
        // Self-attention parameters
        numLayers * (embedDim * embedDim * 4 + embedDim * 2) +
        // FFN parameters
        numLayers * (embedDim * ffnDim * 2 + ffnDim + embedDim) +
        // Layer norm parameters
        numLayers * (embedDim * 4) +
        // Head parameters
        numHeads * embedDim * embedDim
    );
}