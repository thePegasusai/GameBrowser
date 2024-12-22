/**
 * @fileoverview Web Worker implementation for model training operations with memory optimization
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs'; // v4.x
import { DiTModel } from '../lib/model/dit';
import { VAE } from '../lib/model/vae';
import { TensorOperations } from '../lib/tensor/operations';
import { TrainingConfig } from '../config/training';
import { ModelState, MemoryStatus } from '../types/model';

// Worker context type definition
const ctx: Worker = self as any;

// Global state management
let ditModel: DiTModel | null = null;
let vae: VAE | null = null;
let tensorOps: TensorOperations | null = null;
let isTraining: boolean = false;
let currentBatch: number = 0;
let lastMemoryCheck: number = 0;
let trainingConfig: TrainingConfig;

/**
 * Message handler for training operations
 */
ctx.onmessage = async (event: MessageEvent) => {
    try {
        const { type, data } = event.data;

        switch (type) {
            case 'initialize':
                await handleInitialization(data);
                break;
            case 'train':
                await handleTraining(data);
                break;
            case 'cleanup':
                await handleCleanup();
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        handleError(error);
    }
};

/**
 * Handles worker initialization and model setup
 */
async function handleInitialization(data: { 
    config: TrainingConfig, 
    modelWeights?: ArrayBuffer 
}): Promise<void> {
    try {
        // Initialize TensorFlow.js with WebGL backend
        await tf.setBackend('webgl');
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
        tf.env().set('WEBGL_VERSION', 2);

        // Initialize tensor operations
        tensorOps = new TensorOperations(
            data.config.memoryConfig.tensorPoolSize,
            data.config.memoryConfig.disposalStrategy
        );

        // Initialize models
        ditModel = new DiTModel(data.config, tensorOps);
        vae = new VAE(data.config, tensorOps);

        // Load weights if provided
        if (data.modelWeights) {
            await ditModel.loadWeights(data.modelWeights);
        }

        trainingConfig = data.config;

        ctx.postMessage({ type: 'initialized' });
    } catch (error) {
        handleError(error);
    }
}

/**
 * Handles training process with memory optimization
 */
async function handleTraining(data: { 
    frames: Float32Array, 
    actions: Float32Array, 
    batchSize: number 
}): Promise<void> {
    if (!ditModel || !vae || !tensorOps) {
        throw new Error('Models not initialized');
    }

    isTraining = true;
    const startTime = performance.now();

    try {
        // Convert input data to tensors
        const frameTensor = tf.tidy(() => 
            tf.tensor4d(data.frames, [-1, 256, 256, 3])
        );
        const actionTensor = tf.tidy(() => 
            tf.tensor2d(data.actions, [-1, data.actions.length / data.batchSize])
        );

        // Process batch
        const { loss, metrics } = await trainStep(frameTensor, actionTensor);

        // Check memory status
        await checkMemoryStatus();

        // Report progress
        ctx.postMessage({
            type: 'progress',
            data: {
                loss,
                metrics,
                batch: currentBatch++,
                timePerStep: performance.now() - startTime
            }
        });

        // Cleanup tensors
        tf.dispose([frameTensor, actionTensor]);

    } catch (error) {
        isTraining = false;
        handleError(error);
    }
}

/**
 * Executes a single training step with memory optimization
 */
async function trainStep(
    frames: tf.Tensor4D,
    actions: tf.Tensor
): Promise<{ loss: number; metrics: any }> {
    return tf.tidy(() => {
        // Encode frames
        const latents = vae!.encode(frames);

        // Generate noise schedule
        const timeSteps = tf.randomUniform([frames.shape[0]], 0, 1);
        
        // Forward pass through DiT
        const predicted = ditModel!.call(latents, timeSteps, actions);

        // Calculate loss
        const loss = tf.mean(tf.squaredDifference(predicted, latents));

        // Return metrics
        return {
            loss: loss.dataSync()[0],
            metrics: {
                memoryUsage: tf.memory(),
                timeSteps: timeSteps.dataSync()
            }
        };
    });
}

/**
 * Monitors and manages memory usage
 */
async function checkMemoryStatus(): Promise<void> {
    const now = Date.now();
    if (now - lastMemoryCheck < 1000) return;

    const memoryInfo = tf.memory();
    const memoryStatus: MemoryStatus = {
        available: memoryInfo.numBytes,
        required: trainingConfig.memoryConfig.maxTensorAllocation,
        canAllocate: true
    };

    if (memoryInfo.numBytes > trainingConfig.memoryConfig.maxTensorAllocation * 0.9) {
        await tensorOps!.optimizeMemory();
        tf.engine().startScope();
        tf.engine().endScope();
    }

    lastMemoryCheck = now;
    ctx.postMessage({ type: 'memoryStatus', data: memoryStatus });
}

/**
 * Handles cleanup and resource disposal
 */
async function handleCleanup(): Promise<void> {
    isTraining = false;

    try {
        // Dispose models
        ditModel?.dispose();
        vae?.dispose();

        // Clean up tensor operations
        await tensorOps?.cleanup();

        // Force garbage collection
        tf.engine().disposeVariables();
        tf.engine().purgeUnusedTensors();

        ctx.postMessage({ type: 'cleaned' });
    } catch (error) {
        handleError(error);
    }
}

/**
 * Handles and reports errors
 */
function handleError(error: Error): void {
    console.error('Training worker error:', error);
    ctx.postMessage({
        type: 'error',
        data: {
            message: error.message,
            stack: error.stack,
            state: isTraining ? ModelState.TRAINING : ModelState.ERROR
        }
    });
}

// Error event handler
ctx.onerror = (error: ErrorEvent) => {
    handleError(error.error);
};