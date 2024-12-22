/**
 * @fileoverview Core validation utility module for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { memory, ENV } from '@tensorflow/tfjs-core'; // v4.x
import { ModelConfig, ModelType } from '../../types';
import { TensorSpec } from '../../types/tensor';
import { DEFAULT_DIT_CONFIG } from '../../constants/model';
import { DEFAULT_WEBGL_CONFIG } from '../../config/webgl';

// Validation constants
const MAX_TENSOR_DIMENSION = 8192;
const MIN_MEMORY_BUFFER = 512; // MB
const MAX_MEMORY_USAGE = 4096; // MB
const CLEANUP_THRESHOLD = 0.85;
const VALIDATION_TIMEOUT = 5000; // ms

/**
 * Result interfaces for validation operations
 */
interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

interface MemoryValidationResult {
  canAllocate: boolean;
  availableMemory: number;
  requiredMemory: number;
  warnings: string[];
}

interface BrowserValidationResult {
  isCompatible: boolean;
  missingFeatures: string[];
  warnings: string[];
  webglInfo: WebGLInfo;
}

interface WebGLInfo {
  version: string;
  vendor: string;
  renderer: string;
  maxTextureSize: number;
}

/**
 * Validates model configuration parameters against system constraints
 * @param config - Model configuration to validate
 * @returns Detailed validation result
 */
export async function validateModelConfig(config: ModelConfig): Promise<ValidationResult> {
  const result: ValidationResult = {
    isValid: true,
    errors: [],
    warnings: []
  };

  try {
    // Validate model type
    if (!Object.values(ModelType).includes(config.modelType)) {
      result.errors.push(`Invalid model type: ${config.modelType}`);
    }

    // Validate memory constraints
    const memoryResult = await validateMemoryConstraints(
      config.memoryConstraints.maxRAMUsage,
      { gpuMemory: config.memoryConstraints.maxGPUMemory }
    );

    if (!memoryResult.canAllocate) {
      result.errors.push(`Insufficient memory: requires ${memoryResult.requiredMemory}MB`);
    }
    result.warnings.push(...memoryResult.warnings);

    // Validate browser capabilities
    const browserResult = await validateBrowserCapabilities({
      webgl2Required: true,
      checkWebWorker: true
    });

    if (!browserResult.isCompatible) {
      result.errors.push('Browser compatibility check failed');
      result.errors.push(...browserResult.missingFeatures.map(f => `Missing feature: ${f}`));
    }
    result.warnings.push(...browserResult.warnings);

    // Model-specific validations
    switch (config.modelType) {
      case ModelType.DiT:
        validateDiTConfig(config, result);
        break;
      case ModelType.VAE:
        validateVAEConfig(config, result);
        break;
    }

    result.isValid = result.errors.length === 0;
  } catch (error) {
    result.isValid = false;
    result.errors.push(`Validation error: ${error.message}`);
  }

  return result;
}

/**
 * Validates memory constraints with GPU consideration
 * @param requiredMemory - Required memory in MB
 * @param options - Memory validation options
 * @returns Detailed memory validation result
 */
export async function validateMemoryConstraints(
  requiredMemory: number,
  options: { gpuMemory?: number } = {}
): Promise<MemoryValidationResult> {
  const result: MemoryValidationResult = {
    canAllocate: true,
    availableMemory: 0,
    requiredMemory,
    warnings: []
  };

  try {
    // Check total system memory
    const totalMemory = memory().totalMemory / (1024 * 1024); // Convert to MB
    const usedMemory = memory().numBytes / (1024 * 1024);
    result.availableMemory = totalMemory - usedMemory;

    // Validate against maximum memory threshold
    if (requiredMemory > MAX_MEMORY_USAGE) {
      result.canAllocate = false;
      result.warnings.push(`Required memory exceeds maximum limit of ${MAX_MEMORY_USAGE}MB`);
    }

    // Check GPU memory if WebGL backend is active
    if (ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED') && options.gpuMemory) {
      const webglMemInfo = await getWebGLMemoryInfo();
      if (options.gpuMemory > webglMemInfo.maxTextureSize) {
        result.warnings.push('GPU memory requirement exceeds available texture size');
      }
    }

    // Check if cleanup is needed
    if (result.availableMemory < MIN_MEMORY_BUFFER) {
      result.warnings.push('Low memory available, cleanup recommended');
      await memory().dispose();
    }

    // Final allocation check
    result.canAllocate = result.canAllocate && 
                        (result.availableMemory >= requiredMemory + MIN_MEMORY_BUFFER);

  } catch (error) {
    result.canAllocate = false;
    result.warnings.push(`Memory validation error: ${error.message}`);
  }

  return result;
}

/**
 * Validates browser capabilities and features
 * @param options - Browser validation options
 * @returns Detailed browser compatibility result
 */
export async function validateBrowserCapabilities(
  options: { webgl2Required?: boolean; checkWebWorker?: boolean } = {}
): Promise<BrowserValidationResult> {
  const result: BrowserValidationResult = {
    isCompatible: true,
    missingFeatures: [],
    warnings: [],
    webglInfo: await getWebGLInfo()
  };

  try {
    // Validate WebGL 2.0 support
    if (options.webgl2Required && !isWebGL2Supported()) {
      result.isCompatible = false;
      result.missingFeatures.push('WebGL 2.0');
    }

    // Validate Web Workers support
    if (options.checkWebWorker && !isWebWorkerSupported()) {
      result.isCompatible = false;
      result.missingFeatures.push('Web Workers');
    }

    // Check IndexedDB support
    if (!isIndexedDBSupported()) {
      result.warnings.push('IndexedDB not supported, persistence will be limited');
    }

    // Validate WebGL capabilities
    if (result.webglInfo.maxTextureSize < MAX_TENSOR_DIMENSION) {
      result.warnings.push(`Limited texture size: ${result.webglInfo.maxTextureSize}`);
    }

  } catch (error) {
    result.isCompatible = false;
    result.warnings.push(`Browser validation error: ${error.message}`);
  }

  return result;
}

/**
 * Validates tensor operations and specifications
 * @param spec - Tensor specification to validate
 * @param operationType - Type of tensor operation
 * @returns Validation result for tensor operations
 */
export function validateTensorOperations(
  spec: TensorSpec,
  operationType: string
): ValidationResult {
  const result: ValidationResult = {
    isValid: true,
    errors: [],
    warnings: []
  };

  try {
    // Validate tensor dimensions
    if (!spec.shape || !Array.isArray(spec.shape)) {
      result.errors.push('Invalid tensor shape specification');
    } else {
      for (const dim of spec.shape) {
        if (dim > MAX_TENSOR_DIMENSION) {
          result.errors.push(`Tensor dimension ${dim} exceeds maximum ${MAX_TENSOR_DIMENSION}`);
        }
      }
    }

    // Validate data type
    if (!spec.dtype) {
      result.errors.push('Missing tensor data type');
    }

    // Operation-specific validation
    switch (operationType) {
      case 'matmul':
        validateMatmulOperation(spec, result);
        break;
      case 'conv2d':
        validateConv2dOperation(spec, result);
        break;
      default:
        result.warnings.push(`Unknown operation type: ${operationType}`);
    }

    result.isValid = result.errors.length === 0;
  } catch (error) {
    result.isValid = false;
    result.errors.push(`Tensor validation error: ${error.message}`);
  }

  return result;
}

// Helper functions
function validateDiTConfig(config: ModelConfig, result: ValidationResult): void {
  const ditConfig = config.ditConfig || DEFAULT_DIT_CONFIG;
  if (ditConfig.numLayers <= 0) {
    result.errors.push('Invalid number of layers');
  }
  if (ditConfig.hiddenSize <= 0) {
    result.errors.push('Invalid hidden size');
  }
}

function validateVAEConfig(config: ModelConfig, result: ValidationResult): void {
  const vaeConfig = config.vaeConfig;
  if (!vaeConfig) {
    result.errors.push('Missing VAE configuration');
    return;
  }
  if (vaeConfig.latentDim <= 0) {
    result.errors.push('Invalid latent dimension');
  }
}

function validateMatmulOperation(spec: TensorSpec, result: ValidationResult): void {
  if (spec.shape.length !== 2) {
    result.errors.push('Matrix multiplication requires 2D tensors');
  }
}

function validateConv2dOperation(spec: TensorSpec, result: ValidationResult): void {
  if (spec.shape.length !== 4) {
    result.errors.push('Convolution requires 4D tensors [batch, height, width, channels]');
  }
}

async function getWebGLInfo(): Promise<WebGLInfo> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  
  if (!gl) {
    throw new Error('WebGL 2.0 not supported');
  }

  return {
    version: gl.getParameter(gl.VERSION),
    vendor: gl.getParameter(gl.VENDOR),
    renderer: gl.getParameter(gl.RENDERER),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE)
  };
}

async function getWebGLMemoryInfo(): Promise<{ maxTextureSize: number }> {
  const gl = document.createElement('canvas').getContext('webgl2');
  return {
    maxTextureSize: gl ? gl.getParameter(gl.MAX_TEXTURE_SIZE) : 0
  };
}

function isWebGL2Supported(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!canvas.getContext('webgl2');
  } catch {
    return false;
  }
}

function isWebWorkerSupported(): boolean {
  return typeof Worker !== 'undefined';
}

function isIndexedDBSupported(): boolean {
  return typeof indexedDB !== 'undefined';
}