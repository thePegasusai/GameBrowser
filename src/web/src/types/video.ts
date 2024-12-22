/**
 * @fileoverview Type definitions for video processing and frame handling in browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { Tensor, DataType } from '@tensorflow/tfjs-core'; // v4.x
import { TensorSpec } from './tensor';

/**
 * Interface defining structure for individual video frame data
 * Stores frame data as tensor with associated metadata
 */
export interface VideoFrame {
    /** Tensor representation of frame data */
    readonly data: Tensor;
    
    /** Timestamp of frame in milliseconds */
    readonly timestamp: number;
    
    /** Sequential frame index */
    readonly index: number;
}

/**
 * Supported video file formats
 * Limited to browser-compatible formats
 */
export const enum VideoFormat {
    MP4 = 'mp4',
    WEBM = 'webm'
}

/**
 * Interface for video file metadata
 * Contains essential properties for video processing
 */
export interface VideoMetadata {
    /** Video width in pixels */
    readonly width: number;
    
    /** Video height in pixels */
    readonly height: number;
    
    /** Frame rate in frames per second */
    readonly frameRate: number;
    
    /** Total duration in milliseconds */
    readonly duration: number;
    
    /** Video container format */
    readonly format: VideoFormat;
}

/**
 * Configuration interface for video processing pipeline
 * Controls performance and resource usage parameters
 */
export interface VideoProcessingConfig {
    /** Target dimensions [height, width] for processed frames */
    readonly targetSize: readonly number[];
    
    /** Number of frames to skip during processing */
    readonly frameStride: number;
    
    /** Tensor specification for frame data */
    readonly tensorSpec: TensorSpec;
    
    /** Maximum memory usage in bytes */
    readonly maxMemoryUsage: number;
    
    /** Flag to enable WebGL acceleration */
    readonly useWebGL: boolean;
    
    /** Processing timeout in milliseconds */
    readonly processingTimeout: number;
}

/**
 * Enumeration of video processing pipeline states
 * Tracks current status of processing operations
 */
export const enum VideoProcessingState {
    IDLE = 'idle',
    LOADING = 'loading',
    PROCESSING = 'processing',
    COMPLETED = 'completed',
    ERROR = 'error'
}

/**
 * Type definition for video processing errors
 * Provides detailed error information for debugging
 */
export type VideoProcessingError = {
    /** Numeric error code */
    readonly code: number;
    
    /** Human-readable error message */
    readonly message: string;
    
    /** Additional error context */
    readonly details: Record<string, unknown>;
};

/**
 * Type guard for VideoFrame interface
 * @param value - Value to check
 * @returns boolean indicating if value is VideoFrame
 */
export function isVideoFrame(value: unknown): value is VideoFrame {
    return (
        typeof value === 'object' &&
        value !== null &&
        'data' in value &&
        'timestamp' in value &&
        'index' in value &&
        value.data instanceof Tensor &&
        typeof value.timestamp === 'number' &&
        typeof value.index === 'number'
    );
}

/**
 * Type guard for VideoMetadata interface
 * @param value - Value to check
 * @returns boolean indicating if value is VideoMetadata
 */
export function isVideoMetadata(value: unknown): value is VideoMetadata {
    return (
        typeof value === 'object' &&
        value !== null &&
        'width' in value &&
        'height' in value &&
        'frameRate' in value &&
        'duration' in value &&
        'format' in value &&
        typeof value.width === 'number' &&
        typeof value.height === 'number' &&
        typeof value.frameRate === 'number' &&
        typeof value.duration === 'number' &&
        Object.values(VideoFormat).includes(value.format as VideoFormat)
    );
}

/**
 * Type guard for VideoProcessingConfig interface
 * @param value - Value to check
 * @returns boolean indicating if value is VideoProcessingConfig
 */
export function isVideoProcessingConfig(value: unknown): value is VideoProcessingConfig {
    return (
        typeof value === 'object' &&
        value !== null &&
        'targetSize' in value &&
        'frameStride' in value &&
        'tensorSpec' in value &&
        'maxMemoryUsage' in value &&
        'useWebGL' in value &&
        'processingTimeout' in value &&
        Array.isArray(value.targetSize) &&
        value.targetSize.length === 2 &&
        value.targetSize.every(dim => typeof dim === 'number') &&
        typeof value.frameStride === 'number' &&
        typeof value.maxMemoryUsage === 'number' &&
        typeof value.useWebGL === 'boolean' &&
        typeof value.processingTimeout === 'number'
    );
}