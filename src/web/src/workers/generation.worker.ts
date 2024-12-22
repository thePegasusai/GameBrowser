/**
 * @fileoverview Web Worker implementation for video game footage generation using DiT model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import * as comlink from 'comlink'; // v4.x
import { DiTModel } from '../lib/model/dit';
import { VAE } from '../lib/model/vae';
import { TensorOperations } from '../lib/tensor/operations';
import { Logger } from '../lib/utils/logger';
import { validateTensorOperations } from '../lib/utils/validation';
import { ModelState } from '../types/model';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../constants/model';
import { DEFAULT_WEBGL_CONFIG } from '../config/webgl';

/**
 * Worker context and global variables
 */
const ctx = self as unknown as DedicatedWorkerGlobalScope;
let ditModel: DiTModel | null = null;
let vae: VAE | null = null;
let tensorOps: TensorOperations | null = null;
let logger: Logger | null = null;
let isInitialized = false;

/**
 * Generation worker class with memory optimization and WebGL acceleration
 */
class GenerationWorker {
    private readonly performanceMetrics: {
        inferenceTime: number[];
        memoryUsage: number[];
        timestamp: number[];
    };

    constructor() {
        this.performanceMetrics = {
            inferenceTime: [],
            memoryUsage: [],
            timestamp: []
        };

        // Initialize TensorFlow.js with WebGL backend
        this.initializeTensorFlow();
    }

    /**
     * Initializes TensorFlow.js and WebGL context
     * @private
     */
    private async initializeTensorFlow(): Promise<void> {
        try {
            // Set up WebGL backend
            await tf.setBackend('webgl');
            tf.env().set('WEBGL_VERSION', 2);
            tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
            tf.env().set('WEBGL_PACK', true);

            // Initialize components
            logger = new Logger({ 
                level: 'info',
                namespace: 'generation-worker',
                memoryThreshold: MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD
            });

            tensorOps = new TensorOperations(
                new TensorMemoryManager(MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE, 0.9, logger),
                logger
            );

            isInitialized = true;
        } catch (error) {
            logger?.error('Failed to initialize TensorFlow.js:', error);
            throw error;
        }
    }

    /**
     * Initializes models with provided configurations
     * @param ditConfig DiT model configuration
     * @param vaeConfig VAE model configuration
     */
    public async initializeModels(ditConfig: any, vaeConfig: any): Promise<void> {
        try {
            if (!isInitialized || !tensorOps || !logger) {
                throw new Error('Worker not properly initialized');
            }

            ditModel = new DiTModel(ditConfig, tensorOps, logger);
            vae = new VAE(vaeConfig, tensorOps);

            await this.validateModels();
        } catch (error) {
            logger?.error('Failed to initialize models:', error);
            throw error;
        }
    }

    /**
     * Generates a new frame with memory optimization
     * @param inputFrame Input video frame tensor
     * @param actionEmbedding Game action embedding tensor
     * @param timeEmbedding Timestep embedding tensor
     * @returns Generated frame tensor
     */
    public async generateFrame(
        inputFrame: tf.Tensor,
        actionEmbedding: tf.Tensor,
        timeEmbedding: tf.Tensor
    ): Promise<tf.Tensor> {
        const startTime = performance.now();

        try {
            if (!ditModel || !vae || !tensorOps) {
                throw new Error('Models not initialized');
            }

            // Validate input tensors
            const validation = validateTensorOperations(
                { shape: inputFrame.shape, dtype: inputFrame.dtype },
                'generation'
            );
            if (!validation.isValid) {
                throw new Error(`Invalid input tensor: ${validation.errors.join(', ')}`);
            }

            // Process frame with memory optimization
            return await tf.tidy(async () => {
                // Encode input frame
                const latents = await vae.encode(inputFrame as tf.Tensor4D);
                
                // Generate new frame with DiT
                const generatedLatents = await ditModel.call(
                    latents,
                    timeEmbedding,
                    actionEmbedding
                );

                // Decode generated frame
                const generatedFrame = await vae.decode(generatedLatents);

                // Track performance metrics
                const inferenceTime = performance.now() - startTime;
                this.updateMetrics(inferenceTime);

                // Check performance threshold
                if (inferenceTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                    logger?.warn(`Generation exceeded time threshold: ${inferenceTime}ms`);
                }

                return generatedFrame;
            });

        } catch (error) {
            logger?.error('Frame generation failed:', error);
            throw error;
        }
    }

    /**
     * Updates performance metrics
     * @private
     */
    private updateMetrics(inferenceTime: number): void {
        this.performanceMetrics.inferenceTime.push(inferenceTime);
        this.performanceMetrics.memoryUsage.push(tf.memory().numBytes);
        this.performanceMetrics.timestamp.push(Date.now());

        // Keep only recent metrics
        if (this.performanceMetrics.inferenceTime.length > 100) {
            this.performanceMetrics.inferenceTime.shift();
            this.performanceMetrics.memoryUsage.shift();
            this.performanceMetrics.timestamp.shift();
        }
    }

    /**
     * Returns current performance metrics
     */
    public getPerformanceMetrics(): typeof this.performanceMetrics {
        return { ...this.performanceMetrics };
    }

    /**
     * Validates model initialization and performance
     * @private
     */
    private async validateModels(): Promise<void> {
        if (!ditModel || !vae) {
            throw new Error('Models not initialized');
        }

        // Validate model states
        if (ditModel.getState() !== ModelState.READY) {
            throw new Error('DiT model not ready');
        }

        // Validate memory constraints
        const memoryInfo = await tensorOps?.getMemoryInfo();
        if (memoryInfo && memoryInfo.totalBytesUsed > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE) {
            throw new Error('Memory usage exceeds limits');
        }
    }

    /**
     * Cleans up resources and memory
     */
    public async cleanup(): Promise<void> {
        try {
            // Dispose models
            await ditModel?.dispose();
            await vae?.dispose();

            // Clean up tensor operations
            await tensorOps?.cleanup();

            // Force WebGL context cleanup
            const gl = tf.backend().getGPGPUContext().gl;
            gl.getExtension('WEBGL_lose_context')?.loseContext();

            // Reset metrics
            this.performanceMetrics.inferenceTime = [];
            this.performanceMetrics.memoryUsage = [];
            this.performanceMetrics.timestamp = [];

            ditModel = null;
            vae = null;
            tensorOps = null;

        } catch (error) {
            logger?.error('Cleanup failed:', error);
            throw error;
        }
    }
}

// Create and expose worker instance using Comlink
const worker = new GenerationWorker();
export type GenerationWorkerType = typeof worker;
comlink.expose(worker);