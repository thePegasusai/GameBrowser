/**
 * @fileoverview Constants and configurations for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { ModelConfig, ModelType, DiTConfig, VAEConfig, BrowserFeature, TensorSpec } from '../types/model';
import { TensorFormat, DefaultTensorSpec } from '../types/tensor';
import { DataType } from '@tensorflow/tfjs-core'; // v4.x

/**
 * Default DiT model architectures optimized for browser execution
 */
export const MODEL_ARCHITECTURES = {
    DiT_SMALL: {
        numLayers: 12,
        hiddenSize: 384,
        numHeads: 6,
        patchSize: 2,
        intermediateSize: 1536,
        dropoutRate: 0.1
    },
    DiT_BASE: {
        numLayers: 16,
        hiddenSize: 512,
        numHeads: 8,
        patchSize: 2,
        intermediateSize: 2048,
        dropoutRate: 0.1
    },
    DiT_LARGE: {
        numLayers: 24,
        hiddenSize: 768,
        numHeads: 12,
        patchSize: 2,
        intermediateSize: 3072,
        dropoutRate: 0.1
    }
} as const;

/**
 * Default VAE architectures for video encoding/decoding
 */
export const VAE_ARCHITECTURES = {
    VAE_SMALL: {
        latentDim: 256,
        encoderLayers: [32, 64, 128, 256],
        decoderLayers: [256, 128, 64, 32]
    },
    VAE_BASE: {
        latentDim: 512,
        encoderLayers: [64, 128, 256, 512],
        decoderLayers: [512, 256, 128, 64]
    }
} as const;

/**
 * Memory management constraints to maintain <4GB usage
 */
export const MEMORY_CONSTRAINTS = {
    MAX_GPU_MEMORY_USAGE: 3 * 1024, // 3GB in MB
    MAX_TENSOR_BUFFER_SIZE: 1024,    // 1GB in MB
    CLEANUP_THRESHOLD: 0.9,          // 90% memory threshold
    MIN_FREE_MEMORY: 512,            // 512MB minimum free
    TENSOR_DISPOSAL_INTERVAL: 1000,  // 1 second
    MEMORY_CHECK_INTERVAL: 5000      // 5 seconds
} as const;

/**
 * Performance thresholds for <50ms inference
 */
export const PERFORMANCE_THRESHOLDS = {
    MAX_INFERENCE_TIME: 50,        // ms
    MAX_TRAINING_STEP_TIME: 200,   // ms
    FPS_TARGET: 30,
    MEMORY_ALERT_THRESHOLD: 0.85,  // 85% memory usage alert
    PERFORMANCE_CHECK_INTERVAL: 1000, // 1 second
    METRICS_LOGGING_INTERVAL: 5000   // 5 seconds
} as const;

/**
 * Browser compatibility requirements
 */
export const BROWSER_COMPATIBILITY = {
    REQUIRED_WEBGL_VERSION: 2.0,
    MIN_MEMORY_GB: 4,
    REQUIRED_FEATURES: [
        BrowserFeature.WEBGL2,
        BrowserFeature.WEBWORKER,
        BrowserFeature.INDEXEDDB,
        BrowserFeature.SHAREDARRAYBUFFER
    ],
    SUPPORTED_BROWSERS: {
        chrome: '90',
        firefox: '88',
        safari: '14',
        edge: '90'
    }
} as const;

/**
 * Default tensor specifications for model I/O
 */
export const DEFAULT_TENSOR_SPECS = {
    VIDEO_FRAME: new DefaultTensorSpec(
        [-1, 256, 256, 3], // [batch, height, width, channels]
        DataType.float32,
        TensorFormat.NHWC
    ),
    LATENT_SPACE: new DefaultTensorSpec(
        [-1, 32, 32, 16], // [batch, latent_h, latent_w, latent_c]
        DataType.float32,
        TensorFormat.NHWC
    ),
    ACTION_EMBEDDING: new DefaultTensorSpec(
        [-1, 128], // [batch, embedding_dim]
        DataType.float32,
        TensorFormat.NHWC
    )
} as const;

/**
 * Default DiT model configuration
 */
export const DEFAULT_DIT_CONFIG: DiTConfig = {
    ...MODEL_ARCHITECTURES.DiT_BASE,
    inputSpec: DEFAULT_TENSOR_SPECS.LATENT_SPACE,
    outputSpec: DEFAULT_TENSOR_SPECS.LATENT_SPACE
} as const;

/**
 * Default VAE model configuration
 */
export const DEFAULT_VAE_CONFIG: VAEConfig = {
    ...VAE_ARCHITECTURES.VAE_BASE,
    inputSpec: DEFAULT_TENSOR_SPECS.VIDEO_FRAME,
    latentSpec: DEFAULT_TENSOR_SPECS.LATENT_SPACE
} as const;

/**
 * Training hyperparameters
 */
export const TRAINING_PARAMS = {
    DEFAULT_BATCH_SIZE: 4,
    DEFAULT_LEARNING_RATE: 0.001,
    DEFAULT_EPOCHS: 100,
    GRADIENT_CLIP_NORM: 1.0,
    WARMUP_STEPS: 1000,
    EMA_DECAY: 0.9999
} as const;

/**
 * Model validation intervals
 */
export const VALIDATION_INTERVALS = {
    MEMORY_CHECK: MEMORY_CONSTRAINTS.MEMORY_CHECK_INTERVAL,
    PERFORMANCE_CHECK: PERFORMANCE_THRESHOLDS.PERFORMANCE_CHECK_INTERVAL,
    METRICS_LOGGING: PERFORMANCE_THRESHOLDS.METRICS_LOGGING_INTERVAL
} as const;