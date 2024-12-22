/**
 * @fileoverview Advanced video encoder implementation with WebGL acceleration and memory optimization
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { performance } from 'perf_hooks'; // v1.x
import { VideoFrame, VideoProcessingConfig, VideoProcessingState } from '../../types/video';
import { TensorOperations } from '../tensor/operations';
import { Logger } from '../utils/logger';
import { validateTensorOperations } from '../utils/validation';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../constants/model';
import { DEFAULT_WEBGL_CONFIG } from '../../config/webgl';

/**
 * Interface for frame encoding options
 */
interface EncodeOptions {
    targetSize?: [number, number];
    format?: tf.DataType;
    normalize?: boolean;
    useWebGL?: boolean;
}

/**
 * Interface for batch encoding options
 */
interface BatchEncodeOptions extends EncodeOptions {
    batchSize?: number;
    parallelProcessing?: boolean;
}

/**
 * Advanced video encoder with WebGL acceleration and memory management
 */
export class VideoEncoder {
    private readonly tensorOps: TensorOperations;
    private readonly config: VideoProcessingConfig;
    private readonly logger: Logger;
    private readonly frameCache: Map<string, tf.Tensor>;
    private readonly webglContext: WebGLRenderingContext;
    private processingState: VideoProcessingState;
    private lastCleanupTime: number;

    constructor(
        config: VideoProcessingConfig,
        tensorOps: TensorOperations,
        logger: Logger
    ) {
        this.config = config;
        this.tensorOps = tensorOps;
        this.logger = logger;
        this.frameCache = new Map();
        this.processingState = VideoProcessingState.IDLE;
        this.lastCleanupTime = Date.now();

        // Initialize WebGL context
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2', DEFAULT_WEBGL_CONFIG);
        if (!gl) {
            throw new Error('WebGL 2.0 is required for video encoding');
        }
        this.webglContext = gl;

        // Configure memory pressure handling
        if ('onmemorypressure' in window) {
            window.addEventListener('memorypressure', () => this.handleMemoryPressure());
        }
    }

    /**
     * Encodes a single video frame with WebGL acceleration
     * @param videoElement - Source video element
     * @param timestamp - Frame timestamp
     * @param options - Encoding options
     * @returns Encoded video frame with metadata
     */
    public async encodeFrame(
        videoElement: HTMLVideoElement,
        timestamp: number,
        options: EncodeOptions = {}
    ): Promise<VideoFrame> {
        const startTime = performance.now();
        const frameKey = `frame_${timestamp}`;

        try {
            // Check frame cache
            if (this.frameCache.has(frameKey)) {
                return this.createVideoFrame(this.frameCache.get(frameKey)!, timestamp);
            }

            // Update processing state
            this.processingState = VideoProcessingState.PROCESSING;

            // Extract frame using WebGL
            const frameTensor = await this.extractFrameTensor(videoElement, options);

            // Process frame
            const processedTensor = await this.processFrame(frameTensor, options);

            // Cache frame if memory allows
            if (this.shouldCacheFrame()) {
                this.frameCache.set(frameKey, processedTensor);
            }

            // Create video frame object
            const videoFrame = this.createVideoFrame(processedTensor, timestamp);

            // Log performance metrics
            const processingTime = performance.now() - startTime;
            this.logPerformanceMetrics('frame_encode', processingTime, processedTensor);

            // Schedule cleanup if needed
            await this.scheduleCleanup();

            return videoFrame;

        } catch (error) {
            this.processingState = VideoProcessingState.ERROR;
            this.logger.error('Frame encoding error', error);
            throw error;
        }
    }

    /**
     * Batch encodes multiple frames with parallel processing
     * @param videoElement - Source video element
     * @param timestamps - Array of frame timestamps
     * @param options - Batch encoding options
     * @returns Array of encoded video frames
     */
    public async encodeBatch(
        videoElement: HTMLVideoElement,
        timestamps: number[],
        options: BatchEncodeOptions = {}
    ): Promise<VideoFrame[]> {
        const startTime = performance.now();
        const batchSize = options.batchSize || 4;

        try {
            // Validate batch parameters
            if (!timestamps.length) {
                throw new Error('Empty timestamp array');
            }

            // Process frames in batches
            const frames: VideoFrame[] = [];
            for (let i = 0; i < timestamps.length; i += batchSize) {
                const batchTimestamps = timestamps.slice(i, i + batchSize);
                const batchPromises = batchTimestamps.map(timestamp =>
                    this.encodeFrame(videoElement, timestamp, options)
                );

                // Process batch in parallel if enabled
                const batchFrames = options.parallelProcessing ?
                    await Promise.all(batchPromises) :
                    await this.processSequentially(batchPromises);

                frames.push(...batchFrames);

                // Check memory after each batch
                await this.checkMemoryStatus();
            }

            // Log batch performance metrics
            const totalTime = performance.now() - startTime;
            this.logPerformanceMetrics('batch_encode', totalTime, null, frames.length);

            return frames;

        } catch (error) {
            this.processingState = VideoProcessingState.ERROR;
            this.logger.error('Batch encoding error', error);
            throw error;
        }
    }

    /**
     * Releases all resources and cleans up memory
     */
    public async dispose(): Promise<void> {
        try {
            // Dispose all cached tensors
            for (const tensor of this.frameCache.values()) {
                tensor.dispose();
            }
            this.frameCache.clear();

            // Reset state
            this.processingState = VideoProcessingState.IDLE;
            this.lastCleanupTime = Date.now();

            // Force garbage collection if available
            if (window.gc) {
                window.gc();
            }

        } catch (error) {
            this.logger.error('Disposal error', error);
            throw error;
        }
    }

    /**
     * Extracts tensor from video frame using WebGL
     */
    private async extractFrameTensor(
        videoElement: HTMLVideoElement,
        options: EncodeOptions
    ): Promise<tf.Tensor> {
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(videoElement);
            const resized = this.resizeFrame(tensor, options.targetSize);
            return options.normalize ? tf.div(resized, 255) : resized;
        });
    }

    /**
     * Processes frame tensor with optimizations
     */
    private async processFrame(
        tensor: tf.Tensor,
        options: EncodeOptions
    ): Promise<tf.Tensor> {
        const validation = validateTensorOperations(this.config.tensorSpec, 'process');
        if (!validation.isValid) {
            throw new Error(`Invalid tensor operation: ${validation.errors.join(', ')}`);
        }

        return this.tensorOps.batchProcess(
            tensor,
            this.config.tensorSpec,
            DEFAULT_WEBGL_CONFIG
        );
    }

    /**
     * Creates VideoFrame object from tensor
     */
    private createVideoFrame(tensor: tf.Tensor, timestamp: number): VideoFrame {
        return {
            data: tensor,
            timestamp,
            metadata: {
                shape: tensor.shape,
                dtype: tensor.dtype,
                processingTime: performance.now()
            }
        };
    }

    /**
     * Handles memory pressure events
     */
    private async handleMemoryPressure(): Promise<void> {
        this.logger.warn('Memory pressure detected, clearing cache');
        await this.clearCache();
    }

    /**
     * Clears frame cache and releases memory
     */
    private async clearCache(): Promise<void> {
        for (const tensor of this.frameCache.values()) {
            tensor.dispose();
        }
        this.frameCache.clear();
        this.lastCleanupTime = Date.now();
    }

    /**
     * Resizes frame tensor to target dimensions
     */
    private resizeFrame(tensor: tf.Tensor, targetSize?: [number, number]): tf.Tensor {
        if (!targetSize) {
            return tensor;
        }
        return tf.image.resizeBilinear(tensor, targetSize);
    }

    /**
     * Processes promises sequentially
     */
    private async processSequentially<T>(promises: Promise<T>[]): Promise<T[]> {
        const results: T[] = [];
        for (const promise of promises) {
            results.push(await promise);
        }
        return results;
    }

    /**
     * Checks if frame should be cached based on memory usage
     */
    private shouldCacheFrame(): boolean {
        const memoryInfo = tf.memory();
        return memoryInfo.numBytes < MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024;
    }

    /**
     * Schedules cleanup if needed
     */
    private async scheduleCleanup(): Promise<void> {
        const now = Date.now();
        if (now - this.lastCleanupTime > MEMORY_CONSTRAINTS.TENSOR_DISPOSAL_INTERVAL) {
            await this.clearCache();
        }
    }

    /**
     * Checks current memory status
     */
    private async checkMemoryStatus(): Promise<void> {
        const memoryInfo = tf.memory();
        if (memoryInfo.numBytes > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024) {
            await this.clearCache();
        }
    }

    /**
     * Logs performance metrics
     */
    private logPerformanceMetrics(
        operation: string,
        time: number,
        tensor?: tf.Tensor | null,
        batchSize?: number
    ): void {
        this.logger.logPerformance(operation, {
            processingTime: time,
            tensorShape: tensor?.shape,
            memoryUsage: tf.memory(),
            batchSize,
            timestamp: Date.now()
        });

        if (time > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
            this.logger.warn(`${operation} exceeded time threshold: ${time}ms`);
        }
    }
}