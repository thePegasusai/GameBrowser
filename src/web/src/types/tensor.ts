/**
 * @fileoverview Type definitions for tensor operations and memory management
 * @version 1.0.0
 * @license MIT
 */

import { Tensor, DataType } from '@tensorflow/tfjs-core'; // v4.x

/**
 * Supported tensor data formats with validation
 */
export enum TensorFormat {
    NHWC = 'NHWC', // Batch, Height, Width, Channels
    NCHW = 'NCHW'  // Batch, Channels, Height, Width
}

// Format validation namespace
export namespace TensorFormat {
    /**
     * Validates if a given format string is supported
     * @param format - Format string to validate
     * @returns boolean indicating if format is valid
     */
    export function isValid(format: string): format is TensorFormat {
        return Object.values(TensorFormat).includes(format as TensorFormat);
    }
}

/**
 * Immutable type alias for tensor dimensions
 * Ensures shape arrays cannot be modified after creation
 */
export type TensorDimensions = readonly number[];

/**
 * Comprehensive tensor specification interface with validation
 * Used for defining tensor requirements and validating tensor instances
 */
export interface TensorSpec {
    /** Immutable shape specification */
    readonly shape: TensorDimensions;
    
    /** TensorFlow data type */
    readonly dtype: DataType;
    
    /** Data format specification */
    readonly format: TensorFormat;

    /**
     * Validates if a tensor matches this specification
     * @param tensor - Tensor to validate
     * @returns boolean indicating if tensor matches spec
     */
    validate(tensor: Tensor): boolean;
}

/**
 * Detailed memory tracking interface for tensor instances
 * Supports memory optimization and resource management
 */
export interface TensorMemoryInfo {
    /** Size in bytes of tensor data */
    readonly byteSize: number;

    /** Indicates if tensor has been disposed */
    readonly isDisposed: boolean;

    /** Reference count for memory management */
    readonly refCount: number;

    /** Timestamp of last tensor usage */
    readonly lastUsed: number;

    /** GPU memory usage in bytes (if using WebGL backend) */
    readonly gpuMemoryUsage: number;

    /** Memory allocation pool identifier */
    readonly memoryPool: 'webgl' | 'cpu';
}

/**
 * Default implementation of TensorSpec interface
 * Provides standard validation logic
 */
export class DefaultTensorSpec implements TensorSpec {
    constructor(
        public readonly shape: TensorDimensions,
        public readonly dtype: DataType,
        public readonly format: TensorFormat
    ) {}

    /**
     * Validates tensor against specification
     * @param tensor - Tensor to validate
     * @returns boolean indicating if tensor is valid
     */
    validate(tensor: Tensor): boolean {
        // Check if tensor is disposed
        if (tensor.isDisposed) {
            return false;
        }

        // Validate shape
        if (tensor.shape.length !== this.shape.length) {
            return false;
        }
        
        for (let i = 0; i < this.shape.length; i++) {
            // -1 in spec shape indicates dynamic dimension
            if (this.shape[i] !== -1 && this.shape[i] !== tensor.shape[i]) {
                return false;
            }
        }

        // Validate dtype
        if (tensor.dtype !== this.dtype) {
            return false;
        }

        return true;
    }
}

/**
 * Type guard for TensorMemoryInfo interface
 * @param value - Value to check
 * @returns boolean indicating if value is TensorMemoryInfo
 */
export function isTensorMemoryInfo(value: unknown): value is TensorMemoryInfo {
    return (
        typeof value === 'object' &&
        value !== null &&
        'byteSize' in value &&
        'isDisposed' in value &&
        'refCount' in value &&
        'lastUsed' in value &&
        'gpuMemoryUsage' in value &&
        'memoryPool' in value &&
        (value as TensorMemoryInfo).memoryPool in ['webgl', 'cpu']
    );
}

/**
 * Type guard for TensorSpec interface
 * @param value - Value to check
 * @returns boolean indicating if value is TensorSpec
 */
export function isTensorSpec(value: unknown): value is TensorSpec {
    return (
        typeof value === 'object' &&
        value !== null &&
        'shape' in value &&
        'dtype' in value &&
        'format' in value &&
        'validate' in value &&
        typeof (value as TensorSpec).validate === 'function'
    );
}