/**
 * @fileoverview Core video processing module with WebGL acceleration and memory optimization
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { mat4 } from 'gl-matrix'; // v3.x
import { VideoFrameExtractor } from './extractor';
import { VideoEncoder } from './encoder';
import { 
    VideoFrame, 
    VideoProcessingConfig, 
    VideoProcessingState, 
    VideoMetadata 
} from '../../types/video';
import { Logger } from '../utils/logger';
import { validateTensorOperations } from '../utils/validation';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../constants/model';

/**
 * Core video processing class that orchestrates frame extraction, encoding,
 * and processing pipeline with optimized memory management and WebGL acceleration
 */
export class VideoProcessor {
    private readonly extractor: VideoFrameExtractor;
    private readonly encoder: VideoEncoder;
    private readonly config: VideoProcessingConfig;
    private readonly logger: Logger;
    private state: VideoProcessingState;
    private processedFrames: VideoFrame[];
    private glContext: WebGLRenderingContext;
    private transformMatrix: mat4;
    private lastProcessingTime: number;
    private memoryUsage: number;

    /**
     * Initializes video processor with required dependencies and WebGL context
     * @param config - Video processing configuration
     * @param logger - Logger instance for metrics and errors
     */
    constructor(config: VideoProcessingConfig, logger: Logger) {
        this.config = config;
        this.logger = logger;
        this.state = VideoProcessingState.IDLE;
        this.processedFrames = [];
        this.memoryUsage = 0;
        this.lastProcessingTime = 0;

        // Initialize WebGL context
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            throw new Error('WebGL 2.0 is required for video processing');
        }
        this.glContext = gl;
        this.transformMatrix = mat4.create();

        // Initialize components with WebGL context
        this.extractor = new VideoFrameExtractor(this.glContext, logger);
        this.encoder = new VideoEncoder(config, this.glContext, logger);

        // Configure memory pressure handling
        if ('onmemorypressure' in window) {
            window.addEventListener('memorypressure', this.handleMemoryPressure.bind(this));
        }
    }

    /**
     * Process video file through extraction and encoding pipeline with optimized batching
     * @param videoFile - Input video file to process
     * @returns Promise resolving to array of processed frames
     */
    public async processVideo(videoFile: File): Promise<VideoFrame[]> {
        const startTime = performance.now();

        try {
            // Validate input
            if (!videoFile || !videoFile.type.startsWith('video/')) {
                throw new Error('Invalid video file format');
            }

            this.state = VideoProcessingState.LOADING;
            this.logger.info('Starting video processing', { fileName: videoFile.name });

            // Get video metadata
            const metadata = await this.getVideoMetadata(videoFile);
            await this.validateProcessingRequirements(metadata);

            // Calculate optimal batch size based on memory constraints
            const batchSize = this.calculateOptimalBatchSize(metadata);
            this.logger.debug('Calculated batch size', { batchSize });

            this.state = VideoProcessingState.PROCESSING;

            // Process video in batches
            const frames = await this.processBatches(videoFile, metadata, batchSize);

            // Update metrics
            this.lastProcessingTime = performance.now() - startTime;
            this.logProcessingMetrics(frames.length, this.lastProcessingTime);

            return frames;

        } catch (error) {
            this.state = VideoProcessingState.ERROR;
            this.logger.error('Video processing failed', error);
            throw error;
        }
    }

    /**
     * Process a batch of frames with memory optimization
     * @param frames - Array of input frames
     * @param batchSize - Size of processing batch
     * @returns Promise resolving to processed frame batch
     */
    public async processFrameBatch(
        frames: VideoFrame[],
        batchSize: number
    ): Promise<VideoFrame[]> {
        const startTime = performance.now();

        try {
            // Validate tensor operations
            const validation = validateTensorOperations(this.config.tensorSpec, 'batch');
            if (!validation.isValid) {
                throw new Error(`Invalid tensor operation: ${validation.errors.join(', ')}`);
            }

            // Process frames in batches with memory management
            const processedFrames: VideoFrame[] = [];
            for (let i = 0; i < frames.length; i += batchSize) {
                const batch = frames.slice(i, Math.min(i + batchSize, frames.length));
                
                // Extract and encode batch
                const encodedBatch = await this.encoder.encodeBatch(
                    batch,
                    this.config.targetSize,
                    { useWebGL: true }
                );

                processedFrames.push(...encodedBatch);

                // Check memory after each batch
                await this.checkMemoryStatus();
            }

            // Log batch metrics
            this.logBatchMetrics(processedFrames.length, performance.now() - startTime);

            return processedFrames;

        } catch (error) {
            this.logger.error('Batch processing failed', error);
            throw error;
        }
    }

    /**
     * Clean up resources and reset state
     */
    public async dispose(): Promise<void> {
        try {
            // Dispose tensors and clear frames
            this.processedFrames.forEach(frame => {
                if (frame.data instanceof tf.Tensor) {
                    frame.data.dispose();
                }
            });
            this.processedFrames = [];

            // Release WebGL resources
            if (this.glContext) {
                this.glContext.getExtension('WEBGL_lose_context')?.loseContext();
            }

            // Dispose components
            await this.encoder.dispose();
            this.state = VideoProcessingState.IDLE;
            this.memoryUsage = 0;

            this.logger.info('Resources disposed successfully');

        } catch (error) {
            this.logger.error('Disposal error', error);
            throw error;
        }
    }

    /**
     * Private helper methods
     */

    private async getVideoMetadata(videoFile: File): Promise<VideoMetadata> {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoFile);
        await new Promise(resolve => { video.onloadedmetadata = resolve; });
        
        return {
            width: video.videoWidth,
            height: video.videoHeight,
            duration: video.duration * 1000,
            frameRate: video.fps || 30
        };
    }

    private async validateProcessingRequirements(metadata: VideoMetadata): Promise<void> {
        const memoryRequired = this.estimateMemoryRequirement(metadata);
        if (memoryRequired > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE) {
            throw new Error('Video exceeds memory constraints');
        }

        if (metadata.width * metadata.height > this.glContext.getParameter(this.glContext.MAX_TEXTURE_SIZE)) {
            throw new Error('Video dimensions exceed WebGL texture limits');
        }
    }

    private calculateOptimalBatchSize(metadata: VideoMetadata): number {
        const frameSize = metadata.width * metadata.height * 4; // RGBA
        const maxBatchSize = Math.floor(
            (MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.8) / frameSize
        );
        return Math.max(1, Math.min(maxBatchSize, 4)); // Limit between 1 and 4
    }

    private async processBatches(
        videoFile: File,
        metadata: VideoMetadata,
        batchSize: number
    ): Promise<VideoFrame[]> {
        const frames: VideoFrame[] = [];
        const totalFrames = Math.floor(metadata.duration * metadata.frameRate / 1000);

        for (let i = 0; i < totalFrames; i += batchSize) {
            const batch = await this.extractor.extractFrameSequence(
                videoFile,
                i,
                Math.min(batchSize, totalFrames - i)
            );

            const processedBatch = await this.processFrameBatch(batch, batchSize);
            frames.push(...processedBatch);

            // Update progress
            this.logger.debug('Processing progress', {
                processed: frames.length,
                total: totalFrames
            });
        }

        return frames;
    }

    private async checkMemoryStatus(): Promise<void> {
        const memoryInfo = await tf.memory();
        this.memoryUsage = memoryInfo.numBytes;

        if (this.memoryUsage > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.9) {
            await this.cleanupMemory();
        }
    }

    private async cleanupMemory(): Promise<void> {
        await tf.dispose(this.processedFrames.map(frame => frame.data));
        this.processedFrames = [];
        await tf.engine().endScope();
        await tf.engine().startScope();
    }

    private handleMemoryPressure(): void {
        this.logger.warn('Memory pressure detected');
        this.cleanupMemory().catch(error => {
            this.logger.error('Memory cleanup failed', error);
        });
    }

    private logProcessingMetrics(frameCount: number, processingTime: number): void {
        this.logger.logPerformance('video_processing', {
            frameCount,
            processingTime,
            framesPerSecond: frameCount / (processingTime / 1000),
            memoryUsage: this.memoryUsage,
            state: this.state
        });
    }

    private logBatchMetrics(batchSize: number, processingTime: number): void {
        this.logger.logPerformance('batch_processing', {
            batchSize,
            processingTime,
            memoryUsage: this.memoryUsage,
            timePerFrame: processingTime / batchSize
        });
    }

    private estimateMemoryRequirement(metadata: VideoMetadata): number {
        return metadata.width * metadata.height * 4 * 2; // RGBA * 2 for double buffering
    }
}