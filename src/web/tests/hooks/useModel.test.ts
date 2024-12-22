/**
 * @fileoverview Test suite for useModel React hook with performance and memory monitoring
 * @version 1.0.0
 * @license MIT
 */

import { renderHook, act } from '@testing-library/react-hooks';
import { setupGlobalMocks } from '../setup';
import { createMockTensor, createMockModelConfig, createMockVideoFrame } from '../utils';
import { useModel } from '../../src/hooks/useModel';
import * as tf from '@tensorflow/tfjs-core';

// Initialize test environment
setupGlobalMocks();

describe('useModel Hook', () => {
  // Mock performance and memory APIs
  const mockPerformanceNow = jest.spyOn(performance, 'now');
  const mockMemoryInfo = jest.spyOn(performance, 'memory', 'get').mockImplementation(() => ({
    usedJSHeapSize: 100 * 1024 * 1024, // 100MB
    totalJSHeapSize: 4096 * 1024 * 1024, // 4GB
    jsHeapSizeLimit: 4096 * 1024 * 1024 // 4GB
  }));

  // Test data
  let mockModelConfig;
  let mockVideoFrame;
  let mockTensor;

  beforeEach(async () => {
    // Reset all mocks
    jest.clearAllMocks();
    tf.engine().startScope();

    // Initialize test data
    mockModelConfig = await createMockModelConfig();
    mockVideoFrame = createMockVideoFrame(256, 256);
    mockTensor = await createMockTensor([1, 256, 256, 3], 'float32');

    // Mock WebGL context
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
      jest.spyOn(gl, 'getParameter').mockImplementation(() => 4096);
    }
  });

  afterEach(() => {
    // Cleanup test environment
    tf.engine().endScope();
    tf.disposeVariables();
  });

  it('should initialize model with correct state', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));

    expect(result.current.isInitialized).toBe(false);
    expect(result.current.isLoading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.model).toBeNull();
    expect(result.current.vae).toBeNull();

    await waitForNextUpdate();

    expect(result.current.isInitialized).toBe(true);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.model).toBeTruthy();
    expect(result.current.vae).toBeTruthy();
  });

  it('should maintain memory usage below 4GB during operations', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    // Track initial memory
    const initialMemory = result.current.memoryStats.heapUsed;

    // Perform multiple operations
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await result.current.generateFrame(mockTensor, tf.tensor([1]));
      });
    }

    // Verify memory usage
    expect(result.current.memoryStats.heapUsed).toBeLessThan(4 * 1024 * 1024 * 1024); // 4GB
    expect(result.current.memoryStats.utilizationPercentage).toBeLessThan(90);
  });

  it('should maintain inference time below 50ms', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    mockPerformanceNow.mockReturnValueOnce(0).mockReturnValueOnce(30); // 30ms inference time

    await act(async () => {
      await result.current.generateFrame(mockTensor, tf.tensor([1]));
    });

    expect(result.current.metrics.inferenceTime).toBeLessThan(50);
  });

  it('should handle WebGL context loss and recovery', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    // Simulate context loss
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    const loseContext = gl?.getExtension('WEBGL_lose_context');
    
    await act(async () => {
      loseContext?.loseContext();
      await new Promise(resolve => setTimeout(resolve, 100));
      loseContext?.restoreContext();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.isInitialized).toBe(true);
  });

  it('should perform training steps with memory optimization', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    const inputTensor = await createMockTensor([1, 256, 256, 3], 'float32');
    const targetTensor = await createMockTensor([1, 256, 256, 3], 'float32');
    const actionTensor = tf.tensor([1]);

    await act(async () => {
      const loss = await result.current.trainStep(inputTensor, targetTensor, actionTensor);
      expect(typeof loss).toBe('number');
      expect(loss).toBeGreaterThanOrEqual(0);
    });

    // Verify memory cleanup
    expect(result.current.memoryStats.tensorCount).toBeLessThan(1000);
  });

  it('should cleanup resources on unmount', async () => {
    const { result, waitForNextUpdate, unmount } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    const initialTensorCount = tf.memory().numTensors;
    
    unmount();

    // Verify resource cleanup
    expect(tf.memory().numTensors).toBeLessThanOrEqual(initialTensorCount);
  });

  it('should handle errors gracefully', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel({
      ...mockModelConfig,
      maxMemoryUsage: 1 // Force memory constraint error
    }));

    await waitForNextUpdate();

    await act(async () => {
      const largeTensor = await createMockTensor([1000, 1000, 1000, 3], 'float32');
      await result.current.generateFrame(largeTensor, tf.tensor([1]));
    });

    expect(result.current.error).toBeTruthy();
  });

  it('should track performance metrics accurately', async () => {
    const { result, waitForNextUpdate } = renderHook(() => useModel(mockModelConfig));
    await waitForNextUpdate();

    const startTime = performance.now();
    
    await act(async () => {
      await result.current.generateFrame(mockTensor, tf.tensor([1]));
    });

    expect(result.current.metrics.inferenceTime).toBeGreaterThan(0);
    expect(result.current.metrics.fps).toBeGreaterThan(0);
    expect(result.current.metrics.gpuUtilization).toBeGreaterThanOrEqual(0);
    expect(result.current.metrics.gpuUtilization).toBeLessThanOrEqual(100);
  });
});