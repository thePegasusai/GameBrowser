/**
 * @fileoverview Web Worker implementation for video processing with memory optimization
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { VideoProcessor } from '../lib/video/processor';
import { Logger } from '../lib/utils/logger';
import { VideoProcessingConfig, VideoProcessingState } from '../types/video';

// Message types for worker communication
enum WorkerMessageType {
    INIT = 'INIT',
    PROCESS_VIDEO = 'PROCESS_VIDEO',
    PROCESS_FRAME = 'PROCESS_FRAME',
    CLEANUP = 'CLEANUP',
    ERROR = 'ERROR',
    STATUS = 'STATUS'
}

interface WorkerMessage {
    type: WorkerMessageType;
    payload: any;
}

// Worker context
let videoProcessor: VideoProcessor | null = null;
let logger: Logger | null = null;
let isProcessing = false;

/**
 * Initializes the worker with configuration and dependencies
 * @param config Video processing configuration
 */
async function initializeWorker(config: VideoProcessingConfig): Promise<void> {
    try {
        // Initialize TensorFlow.js with WebGL backend
        await tf.setBackend('webgl');
        await tf.ready();

        // Initialize logger with worker-specific namespace
        logger = new Logger({
            level: 'info',
            namespace: 'video-worker',
            persistLogs: false,
            metricsRetentionMs: 3600000
        });

        // Initialize video processor
        videoProcessor = new VideoProcessor(config, logger);

        // Log initialization success
        logger.log('Worker initialized successfully', 'info', {
            backend: tf.getBackend(),
            memoryInfo: await tf.memory()
        });

        // Send initialization success message
        self.postMessage({
            type: WorkerMessageType.INIT,
            payload: { success: true }
        });

    } catch (error) {
        handleError('Worker initialization failed', error);
    }
}

/**
 * Processes a video file with memory-optimized batch processing
 * @param videoFile Video file to process
 */
async function processVideo(videoFile: File): Promise<void> {
    if (!videoProcessor || !logger) {
        throw new Error('Worker not initialized');
    }

    try {
        isProcessing = true;
        const startTime = performance.now();

        // Log processing start
        logger.log('Starting video processing', 'info', {
            fileName: videoFile.name,
            fileSize: videoFile.size,
            startTime
        });

        // Process video with progress updates
        const frames = await videoProcessor.processVideo(videoFile);

        // Send success message with processed frames
        self.postMessage({
            type: WorkerMessageType.PROCESS_VIDEO,
            payload: {
                frames,
                processingTime: performance.now() - startTime,
                memoryUsage: await tf.memory()
            }
        });

    } catch (error) {
        handleError('Video processing failed', error);
    } finally {
        isProcessing = false;
    }
}

/**
 * Processes a single frame with memory optimization
 * @param frame Frame data to process
 * @param timestamp Frame timestamp
 */
async function processFrame(frame: ImageData, timestamp: number): Promise<void> {
    if (!videoProcessor || !logger) {
        throw new Error('Worker not initialized');
    }

    try {
        const startTime = performance.now();

        // Process single frame
        const processedFrame = await videoProcessor.processFrame(frame, timestamp);

        // Send processed frame
        self.postMessage({
            type: WorkerMessageType.PROCESS_FRAME,
            payload: {
                frame: processedFrame,
                timestamp,
                processingTime: performance.now() - startTime
            }
        });

    } catch (error) {
        handleError('Frame processing failed', error);
    }
}

/**
 * Handles cleanup and resource disposal
 */
async function cleanup(): Promise<void> {
    try {
        if (videoProcessor) {
            await videoProcessor.cleanup();
        }

        // Force garbage collection if available
        if ('gc' in self) {
            (self as any).gc();
        }

        // Log cleanup success
        if (logger) {
            logger.log('Worker cleanup completed', 'info', {
                memoryInfo: await tf.memory()
            });
        }

        self.postMessage({
            type: WorkerMessageType.CLEANUP,
            payload: { success: true }
        });

    } catch (error) {
        handleError('Cleanup failed', error);
    }
}

/**
 * Handles worker errors with logging and reporting
 */
function handleError(message: string, error: Error): void {
    const errorDetails = {
        message: error.message,
        stack: error.stack,
        timestamp: Date.now()
    };

    // Log error if logger is available
    if (logger) {
        logger.error(error, { context: message });
    }

    // Send error message to main thread
    self.postMessage({
        type: WorkerMessageType.ERROR,
        payload: {
            message,
            error: errorDetails
        }
    });
}

/**
 * Handles memory pressure events
 */
async function handleMemoryPressure(): Promise<void> {
    if (logger) {
        logger.warn('Memory pressure detected in worker');
    }

    try {
        // Cleanup unused tensors
        await tf.disposeVariables();
        await cleanup();

        // Send status update
        self.postMessage({
            type: WorkerMessageType.STATUS,
            payload: {
                memoryPressure: true,
                memoryInfo: await tf.memory()
            }
        });
    } catch (error) {
        handleError('Memory pressure handling failed', error);
    }
}

// Set up message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
    try {
        switch (event.data.type) {
            case WorkerMessageType.INIT:
                await initializeWorker(event.data.payload);
                break;

            case WorkerMessageType.PROCESS_VIDEO:
                if (!isProcessing) {
                    await processVideo(event.data.payload);
                } else {
                    throw new Error('Video processing already in progress');
                }
                break;

            case WorkerMessageType.PROCESS_FRAME:
                await processFrame(event.data.payload.frame, event.data.payload.timestamp);
                break;

            case WorkerMessageType.CLEANUP:
                await cleanup();
                break;

            default:
                throw new Error(`Unknown message type: ${event.data.type}`);
        }
    } catch (error) {
        handleError('Message handling failed', error as Error);
    }
};

// Set up error handler
self.onerror = (event: ErrorEvent) => {
    handleError('Worker error', new Error(event.message));
};

// Set up memory pressure handler
if ('onmemorypressure' in self) {
    self.addEventListener('memorypressure', () => {
        handleMemoryPressure().catch(error => {
            handleError('Memory pressure handler failed', error);
        });
    });
}