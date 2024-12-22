/**
 * @fileoverview Test suite for training web worker implementation
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import '@testing-library/jest-dom'; // v5.x
import { createMockTensor, createMockModelConfig } from '../utils';
import { setupGlobalMocks } from '../setup';
import { 
    ModelType, 
    ModelState,
    ModelConfig,
    MemoryStatus,
    TensorFormat 
} from '../../src/types';
import { 
    PERFORMANCE_THRESHOLDS,
    MEMORY_CONSTRAINTS,
    TRAINING_PARAMS 
} from '../../src/constants/model';

// Mock worker instance
let mockWorker: Worker & { 
    onmessage?: (e: MessageEvent) => void; 
    postMessage: jest.Mock;
};

// Mock training configuration
const mockTrainingConfig = {
    learningRate: TRAINING_PARAMS.DEFAULT_LEARNING_RATE,
    batchSize: TRAINING_PARAMS.DEFAULT_BATCH_SIZE,
    epochs: TRAINING_PARAMS.DEFAULT_EPOCHS,
    modelType: ModelType.DiT
};

describe('Training Worker Tests', () => {
    beforeEach(async () => {
        // Setup test environment
        setupGlobalMocks();
        
        // Initialize mock worker
        mockWorker = {
            postMessage: jest.fn(),
            terminate: jest.fn()
        } as any;

        // Setup WebGL context with memory tracking
        const gl = document.createElement('canvas').getContext('webgl2');
        expect(gl).toBeTruthy();

        // Initialize performance monitoring
        jest.spyOn(performance, 'now');
        jest.spyOn(tf, 'memory');
    });

    afterEach(() => {
        // Cleanup resources
        jest.clearAllMocks();
        tf.disposeVariables();
        
        // Reset worker state
        if (mockWorker) {
            mockWorker.terminate();
        }
    });

    describe('Training Initialization', () => {
        test('should initialize training environment correctly', async () => {
            // Create mock model configuration
            const modelConfig = await createMockModelConfig();
            
            // Verify WebGL context
            const gl = document.createElement('canvas').getContext('webgl2');
            expect(gl).toBeTruthy();
            
            // Test initialization message
            const initMessage = {
                type: 'init',
                config: modelConfig,
                trainingConfig: mockTrainingConfig
            };
            
            // Send initialization message
            mockWorker.postMessage(initMessage);
            
            // Verify message handling
            expect(mockWorker.postMessage).toHaveBeenCalledTimes(1);
            expect(mockWorker.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({ type: 'init' })
            );
        });

        test('should validate memory constraints during initialization', async () => {
            const memoryInfo = await tf.memory();
            expect(memoryInfo.numBytes).toBeLessThan(MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024);
        });
    });

    describe('Training Step Execution', () => {
        test('should execute training step within time constraints', async () => {
            // Create mock input tensors
            const inputTensor = await createMockTensor([4, 256, 256, 3]);
            const targetTensor = await createMockTensor([4, 256, 256, 3]);
            
            // Start performance timer
            const startTime = performance.now();
            
            // Execute training step
            const trainingMessage = {
                type: 'train',
                input: inputTensor,
                target: targetTensor
            };
            
            mockWorker.postMessage(trainingMessage);
            
            // Verify execution time
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            
            expect(executionTime).toBeLessThan(PERFORMANCE_THRESHOLDS.MAX_TRAINING_STEP_TIME);
        });

        test('should properly manage tensor memory during training', async () => {
            const initialMemory = await tf.memory();
            
            // Execute multiple training steps
            for (let i = 0; i < 5; i++) {
                const inputTensor = await createMockTensor([4, 256, 256, 3]);
                const targetTensor = await createMockTensor([4, 256, 256, 3]);
                
                mockWorker.postMessage({
                    type: 'train',
                    input: inputTensor,
                    target: targetTensor
                });
                
                // Ensure tensors are disposed
                inputTensor.dispose();
                targetTensor.dispose();
            }
            
            const finalMemory = await tf.memory();
            expect(finalMemory.numTensors).toBe(initialMemory.numTensors);
        });
    });

    describe('Memory Management', () => {
        test('should maintain memory usage within constraints', async () => {
            const memoryMonitor = jest.fn();
            let currentMemory: MemoryStatus;
            
            // Monitor memory usage
            const intervalId = setInterval(async () => {
                currentMemory = {
                    available: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
                    required: (await tf.memory()).numBytes / (1024 * 1024),
                    canAllocate: true
                };
                memoryMonitor(currentMemory);
            }, 1000);
            
            // Execute intensive operations
            for (let i = 0; i < 10; i++) {
                const tensor = await createMockTensor([8, 256, 256, 3]);
                await tf.nextFrame(); // Allow GC
                tensor.dispose();
            }
            
            clearInterval(intervalId);
            
            // Verify memory constraints
            expect(memoryMonitor).toHaveBeenCalled();
            const calls = memoryMonitor.mock.calls;
            calls.forEach(([memory]) => {
                expect(memory.required).toBeLessThan(MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE);
            });
        });
    });

    describe('Worker Communication', () => {
        test('should handle worker messages correctly', async () => {
            const messageHandler = jest.fn();
            mockWorker.onmessage = messageHandler;
            
            // Test various message types
            const messages = [
                { type: 'init', config: await createMockModelConfig() },
                { type: 'train', epoch: 1, loss: 0.5 },
                { type: 'error', message: 'Test error' }
            ];
            
            messages.forEach(msg => {
                const event = new MessageEvent('message', { data: msg });
                mockWorker.onmessage?.(event);
            });
            
            expect(messageHandler).toHaveBeenCalledTimes(messages.length);
            expect(messageHandler.mock.calls[0][0].data.type).toBe('init');
        });

        test('should handle worker errors gracefully', async () => {
            const errorHandler = jest.fn();
            mockWorker.onerror = errorHandler;
            
            // Simulate worker error
            const errorEvent = new ErrorEvent('error', {
                message: 'Test error',
                error: new Error('Test error')
            });
            
            mockWorker.onerror?.(errorEvent);
            
            expect(errorHandler).toHaveBeenCalled();
            expect(errorHandler.mock.calls[0][0].message).toBe('Test error');
        });
    });
});