/**
 * @fileoverview React hook for managing video processing state and operations
 * @version 1.0.0
 * @license MIT
 */

import { useState, useCallback, useEffect } from 'react'; // v18.0.0
import { VideoProcessor } from '../lib/video/processor';
import { VideoFrame, VideoProcessingState } from '../types/video';
import { validateMemoryConstraints } from '../lib/utils/validation';

/**
 * Interface for video processing metrics
 */
interface ProcessingMetrics {
  frameRate: number;
  processingTime: number;
  memoryUsage: number;
  gpuUtilization: number;
  lastUpdate: number;
}

/**
 * Interface for memory statistics
 */
interface MemoryStats {
  totalUsed: number;
  gpuUsed: number;
  tensorCount: number;
  lastCleanup: number;
}

/**
 * Default configuration for video processing
 */
const DEFAULT_CONFIG = {
  targetSize: [256, 256] as [number, number],
  frameStride: 1,
  batchSize: 4,
  tensorSpec: {
    shape: [1, 256, 256, 3],
    dtype: 'float32'
  },
  memoryLimit: 3.5e9, // 3.5GB
  cleanupThreshold: 0.8
};

/**
 * React hook for managing video processing with memory optimization
 * @param config - Video processing configuration
 * @returns Video processing state and handlers
 */
export function useVideo(config = DEFAULT_CONFIG) {
  // State management
  const [processor, setProcessor] = useState<VideoProcessor | null>(null);
  const [state, setState] = useState<VideoProcessingState>(VideoProcessingState.IDLE);
  const [frames, setFrames] = useState<VideoFrame[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [metrics, setMetrics] = useState<ProcessingMetrics>({
    frameRate: 0,
    processingTime: 0,
    memoryUsage: 0,
    gpuUtilization: 0,
    lastUpdate: Date.now()
  });
  const [memoryStats, setMemoryStats] = useState<MemoryStats>({
    totalUsed: 0,
    gpuUsed: 0,
    tensorCount: 0,
    lastCleanup: Date.now()
  });

  /**
   * Initialize video processor with error handling
   */
  useEffect(() => {
    const initProcessor = async () => {
      try {
        // Validate memory constraints before initialization
        const memoryValidation = await validateMemoryConstraints(
          config.memoryLimit,
          { gpuMemory: config.memoryLimit * 0.7 }
        );

        if (!memoryValidation.canAllocate) {
          throw new Error('Insufficient memory for video processing');
        }

        const newProcessor = new VideoProcessor(config, console);
        setProcessor(newProcessor);
      } catch (err) {
        setError(err as Error);
        setState(VideoProcessingState.ERROR);
      }
    };

    initProcessor();

    // Cleanup on unmount
    return () => {
      processor?.dispose().catch(console.error);
    };
  }, []);

  /**
   * Process video file with memory optimization
   */
  const processVideo = useCallback(async (file: File): Promise<void> => {
    if (!processor) {
      throw new Error('Video processor not initialized');
    }

    try {
      setState(VideoProcessingState.LOADING);
      const startTime = performance.now();

      // Process video in batches
      const processedFrames = await processor.processVideo(file);
      setFrames(processedFrames);

      // Update metrics
      const endTime = performance.now();
      setMetrics(prev => ({
        ...prev,
        processingTime: endTime - startTime,
        frameRate: processedFrames.length / ((endTime - startTime) / 1000),
        lastUpdate: Date.now()
      }));

      setState(VideoProcessingState.COMPLETED);

    } catch (err) {
      setError(err as Error);
      setState(VideoProcessingState.ERROR);
    }
  }, [processor]);

  /**
   * Process individual frame with memory management
   */
  const processFrame = useCallback(async (frame: VideoFrame): Promise<VideoFrame> => {
    if (!processor) {
      throw new Error('Video processor not initialized');
    }

    try {
      // Check memory status before processing
      const memoryInfo = await processor.getState();
      if (memoryInfo.memoryUsage > config.memoryLimit * config.cleanupThreshold) {
        await processor.optimizeMemory();
      }

      // Process frame
      const processedFrame = await processor.processFrame(frame);

      // Update memory stats
      setMemoryStats(prev => ({
        ...prev,
        totalUsed: memoryInfo.memoryUsage,
        gpuUsed: memoryInfo.gpuMemoryUsage || 0,
        tensorCount: memoryInfo.tensorCount || 0,
        lastCleanup: Date.now()
      }));

      return processedFrame;

    } catch (err) {
      setError(err as Error);
      setState(VideoProcessingState.ERROR);
      throw err;
    }
  }, [processor, config]);

  /**
   * Reset processor state and clear resources
   */
  const reset = useCallback(async (): Promise<void> => {
    try {
      if (processor) {
        await processor.dispose();
        await processor.resetContext();
      }
      setFrames([]);
      setError(null);
      setState(VideoProcessingState.IDLE);
      
      // Reset metrics
      setMetrics({
        frameRate: 0,
        processingTime: 0,
        memoryUsage: 0,
        gpuUtilization: 0,
        lastUpdate: Date.now()
      });
    } catch (err) {
      setError(err as Error);
      setState(VideoProcessingState.ERROR);
    }
  }, [processor]);

  // Monitor memory usage and trigger cleanup if needed
  useEffect(() => {
    const checkMemory = async () => {
      if (processor && state !== VideoProcessingState.ERROR) {
        const memoryInfo = await processor.getState();
        if (memoryInfo.memoryUsage > config.memoryLimit * config.cleanupThreshold) {
          await processor.optimizeMemory();
          setMemoryStats(prev => ({
            ...prev,
            lastCleanup: Date.now()
          }));
        }
      }
    };

    const interval = setInterval(checkMemory, 5000);
    return () => clearInterval(interval);
  }, [processor, state, config]);

  return {
    processVideo,
    processFrame,
    state,
    frames,
    error,
    reset,
    metrics,
    memoryUsage: memoryStats
  };
}