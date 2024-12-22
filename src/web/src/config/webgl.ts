/**
 * @fileoverview WebGL configuration and context management for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs'; // v4.x
import { WebGLContextConfig } from '../types';

/**
 * Enhanced WebGL configuration interface with performance and security settings
 */
export interface WebGLConfig extends WebGLContextConfig {
    readonly alpha: boolean;
    readonly antialias: boolean;
    readonly depth: boolean;
    readonly premultipliedAlpha: boolean;
    readonly preserveDrawingBuffer: boolean;
    readonly stencil: boolean;
    readonly failIfMajorPerformanceCaveat: boolean;
    readonly desynchronized: boolean;
    readonly powerPreference: 'default' | 'high-performance' | 'low-power';
    readonly performanceFlags: {
        readonly enableFloatTextures: boolean;
        readonly useTexturePooling: boolean;
        readonly optimizeMemoryUsage: boolean;
    };
    readonly securitySettings: {
        readonly enableContextLossHandling: boolean;
        readonly validateExtensions: boolean;
        readonly secureCrossOrigin: boolean;
    };
}

/**
 * Default WebGL configuration optimized for video game diffusion model
 */
export const DEFAULT_WEBGL_CONFIG: WebGLConfig = {
    version: 'webgl2',
    alpha: false, // Disable alpha for performance
    antialias: true, // Enable antialiasing for better visual quality
    depth: true, // Enable depth buffer for 3D operations
    premultipliedAlpha: false, // Disable premultiplied alpha for correct blending
    preserveDrawingBuffer: false, // Disable buffer preservation for performance
    stencil: false, // Disable stencil buffer if unused
    failIfMajorPerformanceCaveat: true, // Ensure high performance
    desynchronized: false, // Keep synchronized updates for ML operations
    powerPreference: 'high-performance',
    maxTextureSize: 4096, // Default to 4K texture support
    performanceFlags: {
        enableFloatTextures: true, // Enable float textures for ML operations
        useTexturePooling: true, // Enable texture pooling for memory optimization
        optimizeMemoryUsage: true, // Enable memory optimization
    },
    securitySettings: {
        enableContextLossHandling: true, // Handle context loss gracefully
        validateExtensions: true, // Validate WebGL extensions
        secureCrossOrigin: true, // Enable secure cross-origin isolation
    },
};

/**
 * WebGL capability limits and device tier specifications
 */
export const WEBGL_LIMITS = {
    MIN_TEXTURE_SIZE: 2048,
    MAX_TEXTURE_SIZE: 16384,
    MAX_RENDERBUFFER_SIZE: 16384,
    MAX_VIEWPORT_DIMS: [16384, 16384] as const,
    DEVICE_TIERS: {
        LOW: {
            maxTextureSize: 2048,
            maxBufferSize: 16777216, // 16MB
        },
        MEDIUM: {
            maxTextureSize: 4096,
            maxBufferSize: 67108864, // 64MB
        },
        HIGH: {
            maxTextureSize: 8192,
            maxBufferSize: 268435456, // 256MB
        },
    },
    PRECISION_REQUIREMENTS: {
        vertexPrecision: 'highp' as const,
        fragmentPrecision: 'highp' as const,
        minPrecision: 'mediump' as const,
    },
} as const;

/**
 * Interface for device information used in WebGL configuration
 */
interface DeviceInfo {
    readonly gpuTier: keyof typeof WEBGL_LIMITS.DEVICE_TIERS;
    readonly maxTextureSize: number;
    readonly isWebGL2: boolean;
    readonly vendor: string;
    readonly renderer: string;
}

/**
 * Generates optimized WebGL configuration based on device capabilities
 * @param deviceInfo - Device capability information
 * @returns Optimized WebGL configuration
 */
export function getWebGLConfig(deviceInfo: DeviceInfo): WebGLConfig {
    // Validate WebGL 2.0 support
    if (!deviceInfo.isWebGL2) {
        throw new Error('WebGL 2.0 is required for video game diffusion model');
    }

    // Get device tier specifications
    const tierLimits = WEBGL_LIMITS.DEVICE_TIERS[deviceInfo.gpuTier];

    // Calculate optimal texture size
    const maxTextureSize = Math.min(
        deviceInfo.maxTextureSize,
        tierLimits.maxTextureSize,
        WEBGL_LIMITS.MAX_TEXTURE_SIZE
    );

    // Configure based on device capabilities
    const config: WebGLConfig = {
        ...DEFAULT_WEBGL_CONFIG,
        maxTextureSize,
        performanceFlags: {
            ...DEFAULT_WEBGL_CONFIG.performanceFlags,
            // Adjust texture pooling based on available memory
            useTexturePooling: deviceInfo.gpuTier !== 'LOW',
        },
    };

    // Apply vendor-specific optimizations
    if (deviceInfo.vendor.includes('NVIDIA')) {
        config.powerPreference = 'high-performance';
    } else if (deviceInfo.vendor.includes('Intel')) {
        config.antialias = false; // Disable antialiasing on Intel GPUs
    }

    // Validate final configuration
    validateWebGLConfig(config, deviceInfo);

    return config;
}

/**
 * Validates WebGL configuration against device capabilities
 * @param config - WebGL configuration to validate
 * @param deviceInfo - Device capability information
 * @throws Error if configuration is invalid
 */
function validateWebGLConfig(config: WebGLConfig, deviceInfo: DeviceInfo): void {
    // Validate texture size
    if (config.maxTextureSize > deviceInfo.maxTextureSize) {
        throw new Error(`Texture size ${config.maxTextureSize} exceeds device maximum ${deviceInfo.maxTextureSize}`);
    }

    // Validate WebGL version
    if (config.version !== 'webgl2' && deviceInfo.isWebGL2) {
        throw new Error('WebGL 2.0 is required but not configured');
    }

    // Validate performance settings
    if (config.performanceFlags.enableFloatTextures && !tf.env().getBool('WEBGL_FLOAT_TEXTURES_ENABLED')) {
        throw new Error('Float textures are required but not supported');
    }
}

export default {
    DEFAULT_WEBGL_CONFIG,
    WEBGL_LIMITS,
    getWebGLConfig,
};