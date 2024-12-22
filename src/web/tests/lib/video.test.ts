/**
 * @fileoverview Comprehensive test suite for video processing modules
 * @version 1.0.0
 * @license MIT
 */

import '@testing-library/jest-dom'; // v5.x
import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { VideoProcessor } from '../../src/lib/video/processor';
import { VideoFrameExtractor } from '../../src/lib/video/extractor';
import { VideoEncoder } from '../../src/lib/video/encoder';
import { Logger } from '../../src/lib/utils/logger';
import { TensorOperations } from '../../src/lib/tensor/operations';
import { TensorMemoryManager } from '../../src/lib/tensor/memory';
import { VideoProcessingState, VideoProcessingConfig } from '../../src/types/video';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../src/constants/model';

// Test configuration constants
const TEST_CONFIG: VideoProcessingConfig = {
  targetSize: [256, 256],
  frameStride: 1,
  maxMemoryUsage: 4096,
  processingTimeout: 5000,
  useWebGL: true,
  tensorSpec: {
    shape: [-1, 256, 256, 3],
    dtype: 'float32',
    format: 'NHWC'
  }
};

// Browser compatibility matrix
const BROWSER_MATRIX = {
  chrome: 90,
  firefox: 88,
  safari: 14,
  edge: 90
};

describe('Video Processing Test Suite', () => {
  let videoProcessor: VideoProcessor;
  let frameExtractor: VideoFrameExtractor;
  let videoEncoder: VideoEncoder;
  let logger: Logger;
  let tensorOps: TensorOperations;
  let memoryManager: TensorMemoryManager;
  let testVideo: HTMLVideoElement;

  beforeEach(async () => {
    // Initialize logger with test configuration
    logger = new Logger({
      level: 'debug',
      namespace: 'test',
      persistLogs: false
    });

    // Initialize memory manager
    memoryManager = new TensorMemoryManager(
      MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
      MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD,
      logger
    );

    // Initialize tensor operations
    tensorOps = new TensorOperations(memoryManager, logger);

    // Initialize video processing components
    frameExtractor = new VideoFrameExtractor(tensorOps, logger);
    videoEncoder = new VideoEncoder(TEST_CONFIG, tensorOps, logger);
    videoProcessor = new VideoProcessor(TEST_CONFIG, logger);

    // Setup test video element
    testVideo = document.createElement('video');
    testVideo.width = 640;
    testVideo.height = 480;
    testVideo.src = 'data:video/mp4;base64,...'; // Test video data
    await new Promise(resolve => { testVideo.onloadedmetadata = resolve; });
  });

  afterEach(async () => {
    // Cleanup resources
    await videoProcessor.dispose();
    await videoEncoder.dispose();
    await memoryManager.cleanupUnusedTensors();
    tf.disposeVariables();
  });

  describe('Video Processing Pipeline', () => {
    test('should process video frames within performance threshold', async () => {
      const startTime = performance.now();
      const frames = await videoProcessor.processVideo(new File([], 'test.mp4'));
      const processingTime = performance.now() - startTime;

      expect(processingTime / frames.length).toBeLessThan(PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME);
      expect(frames.length).toBeGreaterThan(0);
      frames.forEach(frame => {
        expect(frame.data instanceof tf.Tensor).toBeTruthy();
        expect(frame.data.shape).toEqual([1, 256, 256, 3]);
      });
    });

    test('should maintain memory usage below threshold during processing', async () => {
      const memoryBefore = tf.memory().numBytes;
      await videoProcessor.processVideo(new File([], 'test.mp4'));
      const memoryAfter = tf.memory().numBytes;
      const memoryDelta = memoryAfter - memoryBefore;

      expect(memoryDelta).toBeLessThan(MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024);
    });

    test('should handle video format variations', async () => {
      const formats = ['mp4', 'webm'];
      for (const format of formats) {
        const result = await videoProcessor.processVideo(new File([], `test.${format}`));
        expect(result).toBeTruthy();
      }
    });
  });

  describe('Frame Extraction', () => {
    test('should extract frames with WebGL acceleration', async () => {
      const frame = await frameExtractor.extractFrame(0);
      expect(frame.data.dtype).toBe('float32');
      expect(frame.timestamp).toBe(0);
      expect(frame.data.shape).toEqual([1, 256, 256, 3]);
    });

    test('should handle WebGL context loss gracefully', async () => {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      const loseContext = gl?.getExtension('WEBGL_lose_context');
      
      if (loseContext) {
        loseContext.loseContext();
        await frameExtractor.handleWebGLContextLoss();
        const frame = await frameExtractor.extractFrame(0);
        expect(frame).toBeTruthy();
      }
    });
  });

  describe('Video Encoding', () => {
    test('should encode frames within memory constraints', async () => {
      const frames = await videoEncoder.encodeBatch(testVideo, [0, 100, 200], {
        batchSize: 2,
        parallelProcessing: true
      });

      expect(frames.length).toBe(3);
      const memoryInfo = await memoryManager.getMemoryInfo();
      expect(memoryInfo.totalBytesUsed).toBeLessThan(MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024);
    });

    test('should optimize batch processing performance', async () => {
      const timestamps = Array.from({ length: 10 }, (_, i) => i * 100);
      const startTime = performance.now();
      
      await videoEncoder.encodeBatch(testVideo, timestamps, {
        batchSize: 4,
        parallelProcessing: true
      });

      const avgTime = (performance.now() - startTime) / timestamps.length;
      expect(avgTime).toBeLessThan(PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME);
    });
  });

  describe('Browser Compatibility', () => {
    test('should validate WebGL capabilities', () => {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      expect(gl).toBeTruthy();
      expect(gl?.getParameter(gl.VERSION)).toMatch(/WebGL 2.0/);
    });

    test('should handle different browser environments', () => {
      const userAgents = {
        chrome: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        firefox: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0'
      };

      for (const [browser, ua] of Object.entries(userAgents)) {
        Object.defineProperty(navigator, 'userAgent', { value: ua, configurable: true });
        expect(videoProcessor.processVideo(new File([], 'test.mp4'))).resolves.toBeTruthy();
      }
    });
  });

  describe('Memory Management', () => {
    test('should cleanup resources after processing', async () => {
      const initialTensors = tf.memory().numTensors;
      await videoProcessor.processVideo(new File([], 'test.mp4'));
      await videoProcessor.dispose();
      expect(tf.memory().numTensors).toBe(initialTensors);
    });

    test('should handle memory pressure events', async () => {
      const memoryPressureEvent = new Event('memorypressure');
      window.dispatchEvent(memoryPressureEvent);
      
      const memoryInfo = await memoryManager.getMemoryInfo();
      expect(memoryInfo.utilizationPercentage).toBeLessThan(MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD * 100);
    });
  });
});