/**
 * @fileoverview Comprehensive test suite for useVideo hook
 * @version 1.0.0
 * @license MIT
 */

import { renderHook, act } from '@testing-library/react-hooks';
import { describe, it, expect, beforeEach, afterEach, jest } from '@testing/jest';
import { useVideo } from '../../src/hooks/useVideo';
import { setupGlobalMocks } from '../setup';
import { createMockVideoFrame } from '../utils';
import { VideoProcessingState } from '../../src/types/video';
import * as tf from '@tensorflow/tfjs-core';

// Test configuration constants
const TEST_CONFIG = {
  targetSize: [256, 256] as [number, number],
  frameStride: 1,
  maxMemoryUsage: 4096, // 4GB in MB
  processingTimeout: 5000
};

const PERFORMANCE_THRESHOLDS = {
  frameProcessingMs: 50,
  memoryLimitMb: 4096,
  gpuUtilization: 0.8
};

describe('useVideo Hook', () => {
  // Setup test environment
  beforeEach(() => {
    setupGlobalMocks();
    jest.useFakeTimers();
    
    // Mock performance.now
    jest.spyOn(performance, 'now')
      .mockImplementation(() => Date.now());
    
    // Mock TensorFlow memory
    jest.spyOn(tf, 'memory')
      .mockReturnValue({
        numBytes: 0,
        numTensors: 0,
        numDataBuffers: 0,
        unreliable: false
      });
  });

  afterEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
  });

  it('should initialize with correct default state', () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));

    expect(result.current.state).toBe(VideoProcessingState.IDLE);
    expect(result.current.frames).toHaveLength(0);
    expect(result.current.error).toBeNull();
    expect(result.current.memoryUsage).toBeDefined();
  });

  it('should process video with performance monitoring', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    const mockFile = new File([''], 'test.mp4', { type: 'video/mp4' });
    const startTime = Date.now();

    await act(async () => {
      await result.current.processVideo(mockFile);
    });

    // Verify processing state transitions
    expect(result.current.state).toBe(VideoProcessingState.COMPLETED);
    
    // Verify performance metrics
    expect(result.current.metrics.processingTime).toBeLessThan(PERFORMANCE_THRESHOLDS.frameProcessingMs);
    expect(result.current.metrics.frameRate).toBeGreaterThan(0);
    expect(result.current.metrics.lastUpdate).toBeGreaterThan(startTime);
  });

  it('should handle frame processing with memory constraints', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    const mockFrame = createMockVideoFrame(256, 256);

    await act(async () => {
      await result.current.processFrame({
        data: tf.tensor(mockFrame.data),
        timestamp: Date.now(),
        index: 0
      });
    });

    // Verify memory usage
    expect(result.current.memoryUsage.totalUsed).toBeLessThan(PERFORMANCE_THRESHOLDS.memoryLimitMb * 1024 * 1024);
    expect(result.current.memoryUsage.gpuUsed).toBeLessThan(PERFORMANCE_THRESHOLDS.memoryLimitMb * 1024 * 1024);
  });

  it('should handle memory pressure and cleanup', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    const initialMemory = result.current.memoryUsage.totalUsed;

    // Simulate memory pressure
    await act(async () => {
      window.dispatchEvent(new Event('memorypressure'));
    });

    // Verify cleanup
    expect(result.current.memoryUsage.totalUsed).toBeLessThan(initialMemory);
    expect(result.current.frames).toHaveLength(0);
  });

  it('should validate browser compatibility', () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));

    // Verify WebGL support
    expect(result.current.state).not.toBe(VideoProcessingState.ERROR);
    
    // Mock WebGL context loss
    act(() => {
      const canvas = document.createElement('canvas');
      canvas.dispatchEvent(new Event('webglcontextlost'));
    });

    expect(result.current.error).toBeDefined();
  });

  it('should handle batch processing efficiently', async () => {
    const { result } = renderHook(() => useVideo({
      ...TEST_CONFIG,
      batchSize: 4
    }));

    const mockFrames = Array.from({ length: 10 }, (_, i) => ({
      data: tf.tensor(createMockVideoFrame(256, 256).data),
      timestamp: Date.now() + i * 33, // ~30fps
      index: i
    }));

    await act(async () => {
      for (const frame of mockFrames) {
        await result.current.processFrame(frame);
      }
    });

    // Verify batch processing performance
    expect(result.current.metrics.processingTime / mockFrames.length)
      .toBeLessThan(PERFORMANCE_THRESHOLDS.frameProcessingMs);
  });

  it('should handle errors gracefully', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    
    // Force error by passing invalid file
    await act(async () => {
      await result.current.processVideo(new File([''], 'test.txt', { type: 'text/plain' }));
    });

    expect(result.current.state).toBe(VideoProcessingState.ERROR);
    expect(result.current.error).toBeDefined();
  });

  it('should reset state and cleanup resources', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    
    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.state).toBe(VideoProcessingState.IDLE);
    expect(result.current.frames).toHaveLength(0);
    expect(result.current.error).toBeNull();
    expect(result.current.metrics.frameRate).toBe(0);
  });

  it('should maintain performance under load', async () => {
    const { result } = renderHook(() => useVideo(TEST_CONFIG));
    const mockFile = new File([''], 'test.mp4', { type: 'video/mp4' });
    
    // Process multiple times to simulate load
    await act(async () => {
      for (let i = 0; i < 5; i++) {
        await result.current.processVideo(mockFile);
      }
    });

    // Verify consistent performance
    expect(result.current.metrics.processingTime)
      .toBeLessThan(PERFORMANCE_THRESHOLDS.frameProcessingMs);
    expect(result.current.memoryUsage.totalUsed)
      .toBeLessThan(PERFORMANCE_THRESHOLDS.memoryLimitMb * 1024 * 1024);
  });
});