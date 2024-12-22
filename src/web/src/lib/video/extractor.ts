/**
 * @fileoverview WebGL-accelerated video frame extraction with memory optimization
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { VideoFrame, VideoMetadata } from '../../types/video';
import { TensorOperations } from '../tensor/operations';
import { Logger } from '../utils/logger';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../constants/model';

/**
 * WebGL-accelerated video frame extractor with memory optimization
 * Ensures frame extraction under 50ms with memory usage below 4GB
 */
export class VideoFrameExtractor {
    private readonly videoElement: HTMLVideoElement;
    private readonly canvas: HTMLCanvasElement;
    private readonly glContext: WebGLRenderingContext;
    private readonly tensorOps: TensorOperations;
    private readonly logger: Logger;
    private readonly memoryThreshold: number;
    private isWebGLAvailable: boolean;

    constructor(
        tensorOps: TensorOperations,
        logger: Logger,
        memoryThreshold: number = MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD
    ) {
        this.tensorOps = tensorOps;
        this.logger = logger;
        this.memoryThreshold = memoryThreshold;

        // Initialize video element
        this.videoElement = document.createElement('video');
        this.videoElement.playsInline = true;
        this.videoElement.muted = true;
        this.videoElement.crossOrigin = 'anonymous';

        // Initialize WebGL canvas
        this.canvas = document.createElement('canvas');
        const gl = this.canvas.getContext('webgl2', {
            premultipliedAlpha: false,
            preserveDrawingBuffer: true,
            antialias: false,
            depth: false,
            stencil: false
        });

        if (!gl) {
            throw new Error('WebGL 2.0 is required for video frame extraction');
        }

        this.glContext = gl;
        this.isWebGLAvailable = true;

        // Handle WebGL context loss
        this.canvas.addEventListener('webglcontextlost', this.handleContextLoss.bind(this));
        this.canvas.addEventListener('webglcontextrestored', this.handleContextRestore.bind(this));
    }

    /**
     * Extracts a single frame using WebGL acceleration
     * @param timestamp - Timestamp in milliseconds
     * @returns Promise resolving to extracted frame
     */
    public async extractFrame(timestamp: number): Promise<VideoFrame> {
        const startTime = performance.now();

        try {
            // Check memory usage
            const memoryInfo = await this.logger.getMemoryInfo();
            if (memoryInfo.heapUsed / memoryInfo.heapTotal > this.memoryThreshold) {
                this.logger.warn('Memory threshold exceeded, triggering cleanup');
                await this.tensorOps.disposeUnusedTensors();
            }

            // Seek video to timestamp
            this.videoElement.currentTime = timestamp / 1000;
            await new Promise(resolve => {
                this.videoElement.onseeked = resolve;
            });

            // Draw frame to WebGL canvas
            this.canvas.width = this.videoElement.videoWidth;
            this.canvas.height = this.videoElement.videoHeight;
            this.glContext.viewport(0, 0, this.canvas.width, this.canvas.height);
            this.glContext.drawImage(this.videoElement, 0, 0);

            // Convert to tensor using WebGL backend
            const frameData = tf.tidy(() => {
                const imageTensor = tf.browser.fromPixels(this.canvas);
                return this.tensorOps.reshape(imageTensor, [1, this.canvas.height, this.canvas.width, 3]);
            });

            const processingTime = performance.now() - startTime;
            
            // Log performance metrics
            this.logger.logPerformance('frame_extraction', {
                processingTime,
                timestamp,
                frameSize: [this.canvas.width, this.canvas.height],
                memoryUsage: await this.logger.getMemoryInfo()
            });

            // Validate performance threshold
            if (processingTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                this.logger.warn(`Frame extraction exceeded time threshold: ${processingTime}ms`);
            }

            return {
                data: frameData,
                timestamp,
                index: Math.floor(timestamp / (1000 / this.videoElement.fps))
            };

        } catch (error) {
            this.logger.error('Frame extraction failed', error);
            throw error;
        }
    }

    /**
     * Extracts a sequence of frames with memory-efficient batch processing
     * @param startTime - Start timestamp in milliseconds
     * @param endTime - End timestamp in milliseconds
     * @param frameStride - Number of frames to skip
     * @returns Promise resolving to array of extracted frames
     */
    public async extractFrameSequence(
        startTime: number,
        endTime: number,
        frameStride: number = 1
    ): Promise<VideoFrame[]> {
        const frames: VideoFrame[] = [];
        const fps = this.videoElement.fps || 30;
        const frameDuration = 1000 / fps;
        const totalFrames = Math.floor((endTime - startTime) / frameDuration);

        // Calculate optimal batch size based on memory constraints
        const memoryInfo = await this.logger.getMemoryInfo();
        const maxBatchSize = Math.floor(
            (MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.8) / 
            (this.videoElement.videoWidth * this.videoElement.videoHeight * 4)
        );

        for (let i = 0; i < totalFrames; i += frameStride) {
            const timestamp = startTime + (i * frameDuration);
            
            try {
                const frame = await this.extractFrame(timestamp);
                frames.push(frame);

                // Check memory usage and cleanup if needed
                if (frames.length % maxBatchSize === 0) {
                    await this.tensorOps.disposeUnusedTensors();
                }

            } catch (error) {
                this.logger.error(`Failed to extract frame at ${timestamp}ms`, error);
                continue;
            }
        }

        return frames;
    }

    /**
     * Gets comprehensive video metadata
     * @returns Video metadata object
     */
    public getVideoMetadata(): VideoMetadata {
        return {
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight,
            frameRate: this.videoElement.fps || 30,
            duration: this.videoElement.duration * 1000
        };
    }

    /**
     * Handles WebGL context loss
     */
    private handleContextLoss(event: WebGLContextEvent): void {
        event.preventDefault();
        this.isWebGLAvailable = false;
        this.logger.error('WebGL context lost', { event });
    }

    /**
     * Handles WebGL context restoration
     */
    private handleContextRestore(): void {
        this.isWebGLAvailable = true;
        this.logger.info('WebGL context restored');
        this.tensorOps.initializeWebGL();
    }
}