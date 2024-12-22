/**
 * @fileoverview Test utility functions for browser-based video game diffusion model testing
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import '@testing-library/jest-dom'; // v5.x
import { ModelConfig, ModelType } from '../src/types';
import { validateModelConfig } from '../src/lib/utils/validation';
import { TensorFormat, DefaultTensorSpec } from '../src/types/tensor';

/**
 * Memory tracking options for test tensors
 */
interface MemoryOptions {
  trackUsage: boolean;
  maxMemoryMB: number;
  disposeAfterTest: boolean;
  checkLeaks: boolean;
}

/**
 * Browser simulation options for testing
 */
interface BrowserOptions {
  webgl2Enabled: boolean;
  vendor: string;
  maxTextureSize: number;
  memoryLimitMB: number;
}

/**
 * WebGL testing configuration
 */
interface WebGLOptions {
  version: 'webgl2';
  floatTexturesEnabled: boolean;
  maxTextureSize: number;
  vendor: string;
  renderer: string;
}

// Default test configurations
const DEFAULT_TENSOR_DIMS = [1, 256, 256, 3];
const DEFAULT_MODEL_CONFIG: ModelConfig = {
  modelType: ModelType.DiT,
  batchSize: 1,
  memoryConstraints: {
    maxRAMUsage: 4 * 1024 * 1024 * 1024, // 4GB
    maxGPUMemory: 2 * 1024 * 1024 * 1024, // 2GB
    tensorBufferSize: 256 * 1024 * 1024,   // 256MB
    enableMemoryTracking: true
  },
  browserCapabilities: {
    requiredFeatures: ['WEBGL2', 'WEBWORKER', 'INDEXEDDB'],
    webgl: {
      version: 2,
      maxTextureSize: 4096,
      maxRenderBufferSize: 4096,
      vendor: 'Test',
      renderer: 'Test'
    },
    isCompatible: true
  },
  validation: {
    validate: async () => true,
    checkMemory: async () => ({ available: 4096, required: 2048, canAllocate: true }),
    validateBrowser: async () => ({ supported: true, missingFeatures: [], warnings: [] })
  }
};

/**
 * Creates a mock tensor with memory tracking and WebGL support validation
 * @param dimensions - Tensor dimensions
 * @param dataType - Tensor data type
 * @param options - Memory tracking options
 * @returns Mock tensor with memory tracking
 */
export async function createMockTensor(
  dimensions: number[] = DEFAULT_TENSOR_DIMS,
  dataType: tf.DataType = 'float32',
  options: MemoryOptions = {
    trackUsage: true,
    maxMemoryMB: 4096,
    disposeAfterTest: true,
    checkLeaks: true
  }
): Promise<tf.Tensor> {
  // Track initial memory state
  const initialMemory = tf.memory();

  // Validate dimensions and memory constraints
  const totalElements = dimensions.reduce((a, b) => a * b, 1);
  const estimatedMemoryMB = (totalElements * (dataType === 'float32' ? 4 : 1)) / (1024 * 1024);

  if (estimatedMemoryMB > options.maxMemoryMB) {
    throw new Error(`Tensor would exceed memory limit: ${estimatedMemoryMB}MB > ${options.maxMemoryMB}MB`);
  }

  // Create mock data
  const data = new Float32Array(totalElements).fill(0).map(() => Math.random());
  
  // Create tensor with memory tracking
  const tensor = tf.tensor(data, dimensions, dataType);

  if (options.trackUsage) {
    const finalMemory = tf.memory();
    console.log('Memory delta:', {
      bytes: finalMemory.numBytes - initialMemory.numBytes,
      tensors: finalMemory.numTensors - initialMemory.numTensors
    });
  }

  // Setup automatic cleanup
  if (options.disposeAfterTest) {
    afterEach(() => {
      tensor.dispose();
      if (options.checkLeaks) {
        const leakedTensors = tf.memory().numTensors - initialMemory.numTensors;
        expect(leakedTensors).toBe(0);
      }
    });
  }

  return tensor;
}

/**
 * Creates mock model configuration with browser compatibility validation
 * @param overrides - Configuration overrides
 * @param browserOptions - Browser simulation options
 * @returns Validated mock model configuration
 */
export async function createMockModelConfig(
  overrides: Partial<ModelConfig> = {},
  browserOptions: BrowserOptions = {
    webgl2Enabled: true,
    vendor: 'Test',
    maxTextureSize: 4096,
    memoryLimitMB: 4096
  }
): Promise<ModelConfig> {
  // Create base configuration
  const config: ModelConfig = {
    ...DEFAULT_MODEL_CONFIG,
    ...overrides,
    browserCapabilities: {
      ...DEFAULT_MODEL_CONFIG.browserCapabilities,
      webgl: {
        version: browserOptions.webgl2Enabled ? 2 : 1,
        maxTextureSize: browserOptions.maxTextureSize,
        maxRenderBufferSize: browserOptions.maxTextureSize,
        vendor: browserOptions.vendor,
        renderer: 'Test Renderer'
      }
    }
  };

  // Validate configuration
  const validationResult = await validateModelConfig(config);
  if (!validationResult.isValid) {
    throw new Error(`Invalid model configuration: ${validationResult.errors.join(', ')}`);
  }

  return config;
}

/**
 * Creates mock video frame data with WebGL acceleration
 * @param width - Frame width
 * @param height - Frame height
 * @param options - WebGL configuration options
 * @returns Optimized mock frame data
 */
export function createMockVideoFrame(
  width: number = 256,
  height: number = 256,
  options: WebGLOptions = {
    version: 'webgl2',
    floatTexturesEnabled: true,
    maxTextureSize: 4096,
    vendor: 'Test',
    renderer: 'Test'
  }
): ImageData {
  // Validate dimensions
  if (width > options.maxTextureSize || height > options.maxTextureSize) {
    throw new Error(`Frame dimensions exceed maximum texture size: ${options.maxTextureSize}`);
  }

  // Create canvas for WebGL operations
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  
  const gl = canvas.getContext('webgl2');
  if (!gl && options.version === 'webgl2') {
    throw new Error('WebGL 2 context creation failed');
  }

  // Generate random pixel data
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i] = Math.random() * 255;     // R
    pixels[i + 1] = Math.random() * 255; // G
    pixels[i + 2] = Math.random() * 255; // B
    pixels[i + 3] = 255;                 // A
  }

  return new ImageData(pixels, width, height);
}

/**
 * Creates mock WebGL context with performance monitoring
 * @param options - WebGL context options
 * @returns Instrumented mock WebGL context
 */
export function mockWebGLContext(
  options: WebGLOptions = {
    version: 'webgl2',
    floatTexturesEnabled: true,
    maxTextureSize: 4096,
    vendor: 'Test',
    renderer: 'Test'
  }
): WebGLRenderingContext {
  // Create canvas element
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');

  if (!gl) {
    throw new Error('WebGL 2 context creation failed');
  }

  // Mock WebGL capabilities
  jest.spyOn(gl, 'getParameter').mockImplementation((param) => {
    switch (param) {
      case gl.VERSION:
        return 'WebGL 2.0';
      case gl.VENDOR:
        return options.vendor;
      case gl.RENDERER:
        return options.renderer;
      case gl.MAX_TEXTURE_SIZE:
        return options.maxTextureSize;
      default:
        return null;
    }
  });

  // Monitor WebGL operations
  const monitoredGL = new Proxy(gl, {
    get: (target, prop) => {
      const value = target[prop];
      if (typeof value === 'function') {
        return (...args: any[]) => {
          const result = value.apply(target, args);
          // Log WebGL operations for debugging
          console.debug(`WebGL operation: ${String(prop)}`, { args, result });
          return result;
        };
      }
      return value;
    }
  });

  return monitoredGL;
}

// Export test constants
export const TEST_CONSTANTS = {
  DEFAULT_TENSOR_DIMS,
  DEFAULT_MODEL_CONFIG,
  DEFAULT_FRAME_SIZE: { width: 256, height: 256 }
};