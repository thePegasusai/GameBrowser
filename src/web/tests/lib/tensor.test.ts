/**
 * @fileoverview Test suite for tensor operations and memory management
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { describe, it, expect, beforeEach, afterEach, jest } from 'jest'; // v29.x
import { performance } from 'perf_hooks'; // v1.x

import { TensorMemoryManager } from '../../src/lib/tensor/memory';
import { TensorOperations } from '../../src/lib/tensor/operations';
import { createMockTensor } from '../utils';
import { setupTensorFlowBackend } from '../setup';
import { Logger } from '../../src/lib/utils/logger';
import { TensorFormat, DefaultTensorSpec } from '../../src/types/tensor';

// Constants for testing
const TEST_TENSOR_DIMS = [1, 256, 256, 3];
const MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024; // 4GB
const PERFORMANCE_THRESHOLD_MS = 50;
const WEBGL_MEMORY_LIMIT = 2 * 1024 * 1024 * 1024; // 2GB

describe('TensorMemoryManager', () => {
    let memoryManager: TensorMemoryManager;
    let logger: Logger;

    beforeEach(() => {
        logger = new Logger({
            level: 'debug',
            persistLogs: false,
            namespace: 'tensor-test'
        });
        memoryManager = new TensorMemoryManager(MAX_MEMORY_BYTES, 0.85, logger);
        setupTensorFlowBackend();
    });

    afterEach(() => {
        // Cleanup tensors after each test
        tf.disposeVariables();
        jest.clearAllMocks();
    });

    it('should track tensor memory usage correctly', async () => {
        const tensor = await createMockTensor(TEST_TENSOR_DIMS);
        const tensorId = memoryManager.trackTensor(tensor, {
            shape: TEST_TENSOR_DIMS,
            dtype: 'float32',
            format: TensorFormat.NHWC
        }, 1);

        const memoryInfo = memoryManager.getMemoryInfo();
        expect(memoryInfo.totalBytesUsed).toBeGreaterThan(0);
        expect(memoryInfo.numTensors).toBe(1);
        expect(memoryInfo.utilizationPercentage).toBeLessThan(100);

        memoryManager.disposeTensor(tensorId);
        const finalMemoryInfo = memoryManager.getMemoryInfo();
        expect(finalMemoryInfo.totalBytesUsed).toBe(0);
    });

    it('should handle memory pressure scenarios', async () => {
        const largeTensors = await Promise.all(
            Array(10).fill(null).map(() => createMockTensor([1, 512, 512, 3]))
        );

        const tensorIds = largeTensors.map(tensor => 
            memoryManager.trackTensor(tensor, {
                shape: [1, 512, 512, 3],
                dtype: 'float32',
                format: TensorFormat.NHWC
            }, 1)
        );

        const memoryInfo = memoryManager.getMemoryInfo();
        expect(memoryInfo.utilizationPercentage).toBeLessThan(85);

        // Cleanup
        tensorIds.forEach(id => memoryManager.disposeTensor(id));
    });

    it('should enforce memory limits', async () => {
        const oversizedTensor = await createMockTensor([1, 2048, 2048, 3]);
        
        expect(() => {
            for (let i = 0; i < 100; i++) {
                memoryManager.trackTensor(oversizedTensor, {
                    shape: [1, 2048, 2048, 3],
                    dtype: 'float32',
                    format: TensorFormat.NHWC
                }, 1);
            }
        }).toThrow(/Memory threshold exceeded/);
    });
});

describe('TensorOperations', () => {
    let tensorOps: TensorOperations;
    let memoryManager: TensorMemoryManager;
    let logger: Logger;

    beforeEach(() => {
        logger = new Logger({
            level: 'debug',
            persistLogs: false,
            namespace: 'tensor-test'
        });
        memoryManager = new TensorMemoryManager(MAX_MEMORY_BYTES, 0.85, logger);
        tensorOps = new TensorOperations(memoryManager, logger);
        setupTensorFlowBackend();
    });

    afterEach(() => {
        tf.disposeVariables();
        jest.clearAllMocks();
    });

    it('should perform batch processing within time threshold', async () => {
        const input = await createMockTensor(TEST_TENSOR_DIMS);
        const outputSpec = new DefaultTensorSpec(
            TEST_TENSOR_DIMS,
            'float32',
            TensorFormat.NHWC
        );

        const startTime = performance.now();
        const result = await tensorOps.batchProcess(input, outputSpec, {
            version: 'webgl2',
            maxTextureSize: 4096,
            performanceFlags: {
                enableFloatTextures: true,
                useTexturePooling: true,
                optimizeMemoryUsage: true
            }
        });

        const processingTime = performance.now() - startTime;
        expect(processingTime).toBeLessThan(PERFORMANCE_THRESHOLD_MS);
        expect(result.shape).toEqual(TEST_TENSOR_DIMS);
    });

    it('should handle WebGL context loss gracefully', async () => {
        const input = await createMockTensor(TEST_TENSOR_DIMS);
        const outputSpec = new DefaultTensorSpec(
            TEST_TENSOR_DIMS,
            'float32',
            TensorFormat.NHWC
        );

        // Simulate WebGL context loss
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        gl?.getExtension('WEBGL_lose_context')?.loseContext();

        await expect(async () => {
            await tensorOps.batchProcess(input, outputSpec, {
                version: 'webgl2',
                maxTextureSize: 4096,
                performanceFlags: {
                    enableFloatTextures: true,
                    useTexturePooling: true,
                    optimizeMemoryUsage: true
                }
            });
        }).not.toThrow();
    });

    it('should optimize memory usage during operations', async () => {
        const initialMemory = tf.memory();
        const input = await createMockTensor(TEST_TENSOR_DIMS);

        const result = await tensorOps.reshape(input, [1, -1, 3], {
            version: 'webgl2',
            maxTextureSize: 4096,
            performanceFlags: {
                enableFloatTextures: true,
                useTexturePooling: true,
                optimizeMemoryUsage: true
            }
        });

        const finalMemory = tf.memory();
        expect(finalMemory.numBytes - initialMemory.numBytes).toBeLessThan(
            WEBGL_MEMORY_LIMIT
        );
        expect(result.shape[1]).toBe(256 * 256);
    });

    it('should validate tensor operations', async () => {
        const input = await createMockTensor(TEST_TENSOR_DIMS);
        const invalidSpec = new DefaultTensorSpec(
            [1, 8192, 8192, 3], // Exceeds typical WebGL limits
            'float32',
            TensorFormat.NHWC
        );

        await expect(async () => {
            await tensorOps.batchProcess(input, invalidSpec, {
                version: 'webgl2',
                maxTextureSize: 4096,
                performanceFlags: {
                    enableFloatTextures: true,
                    useTexturePooling: true,
                    optimizeMemoryUsage: true
                }
            });
        }).toThrow(/Invalid tensor operation/);
    });
});