/**
 * @fileoverview React hook for managing browser-based video game diffusion model lifecycle
 * @version 1.0.0
 * @license MIT
 */

import { useState, useEffect, useCallback } from 'react'; // v18.0.0
import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { DiTModel } from '../lib/model/dit';
import { VAE } from '../lib/model/vae';
import { useWebGL } from './useWebGL';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../constants/model';

/**
 * Interface for memory statistics tracking
 */
interface MemoryStats {
  tensorCount: number;
  heapUsed: number;
  gpuMemory: number;
  utilizationPercentage: number;
}

/**
 * Interface for performance metrics
 */
interface PerformanceMetrics {
  inferenceTime: number;
  trainingTime: number;
  fps: number;
  gpuUtilization: number;
}

/**
 * Interface for model configuration
 */
interface ModelConfig {
  batchSize?: number;
  learningRate?: number;
  maxMemoryUsage?: number;
  useWebGL?: boolean;
}

/**
 * Enhanced React hook for managing video game diffusion model lifecycle
 * with optimized memory management and performance monitoring
 */
export function useModel(config: ModelConfig = {}) {
  // Model state management
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [model, setModel] = useState<DiTModel | null>(null);
  const [vae, setVAE] = useState<VAE | null>(null);

  // Performance monitoring
  const [memoryStats, setMemoryStats] = useState<MemoryStats>({
    tensorCount: 0,
    heapUsed: 0,
    gpuMemory: 0,
    utilizationPercentage: 0
  });
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    inferenceTime: 0,
    trainingTime: 0,
    fps: 0,
    gpuUtilization: 0
  });

  // WebGL context management
  const { gl, isContextLost, restoreContext } = useWebGL();

  /**
   * Initializes model with memory optimization
   */
  const initializeModel = useCallback(async () => {
    if (isInitialized || isLoading) return;
    setIsLoading(true);

    try {
      // Configure TensorFlow.js for optimal performance
      await tf.ready();
      tf.engine().startScope();
      tf.setBackend('webgl');
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
      tf.env().set('WEBGL_VERSION', 2);
      tf.env().set('WEBGL_PACK', true);

      // Initialize models with memory tracking
      const tensorOps = new TensorOperations(
        new TensorMemoryManager(config.maxMemoryUsage || MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE),
        new Logger({ level: 'info', namespace: 'model' })
      );

      const ditModel = new DiTModel({
        ...MODEL_ARCHITECTURES.DiT_BASE,
        maxMemoryUsage: config.maxMemoryUsage
      }, tensorOps, new Logger({ level: 'info', namespace: 'dit' }));

      const vaeModel = new VAE({
        ...VAE_ARCHITECTURES.VAE_BASE,
        maxMemoryUsage: config.maxMemoryUsage
      }, tensorOps);

      setModel(ditModel);
      setVAE(vaeModel);
      setIsInitialized(true);

    } catch (err) {
      setError(err as Error);
      console.error('Model initialization failed:', err);
    } finally {
      setIsLoading(false);
      tf.engine().endScope();
    }
  }, [config.maxMemoryUsage, isInitialized, isLoading]);

  /**
   * Generates a new frame with performance monitoring
   */
  const generateFrame = useCallback(async (
    input: tf.Tensor4D,
    actionEmbedding: tf.Tensor
  ): Promise<tf.Tensor4D | null> => {
    if (!model || !vae || isContextLost) return null;

    const startTime = performance.now();
    let result: tf.Tensor4D | null = null;

    try {
      tf.engine().startScope();

      // Encode input frame
      const latents = await vae.encode(input);

      // Generate new frame
      const generated = await model.call(
        latents,
        tf.randomNormal([1, model.config.hiddenSize]),
        actionEmbedding
      );

      // Decode result
      result = await vae.decode(generated) as tf.Tensor4D;

      // Update performance metrics
      const inferenceTime = performance.now() - startTime;
      setMetrics(prev => ({
        ...prev,
        inferenceTime,
        fps: 1000 / inferenceTime
      }));

      return result;

    } catch (err) {
      setError(err as Error);
      console.error('Frame generation failed:', err);
      return null;
    } finally {
      tf.engine().endScope();
    }
  }, [model, vae, isContextLost]);

  /**
   * Performs a training step with memory optimization
   */
  const trainStep = useCallback(async (
    input: tf.Tensor4D,
    target: tf.Tensor4D,
    actionEmbedding: tf.Tensor
  ): Promise<number> => {
    if (!model || !vae || isContextLost) return 0;

    const startTime = performance.now();
    let loss = 0;

    try {
      tf.engine().startScope();

      // Encode input and target
      const [inputLatents, targetLatents] = await Promise.all([
        vae.encode(input),
        vae.encode(target)
      ]);

      // Train model
      const generated = await model.call(
        inputLatents,
        tf.randomNormal([1, model.config.hiddenSize]),
        actionEmbedding
      );

      loss = tf.losses.meanSquaredError(targetLatents, generated).dataSync()[0];

      // Update performance metrics
      const trainingTime = performance.now() - startTime;
      setMetrics(prev => ({
        ...prev,
        trainingTime
      }));

      return loss;

    } catch (err) {
      setError(err as Error);
      console.error('Training step failed:', err);
      return 0;
    } finally {
      tf.engine().endScope();
    }
  }, [model, vae, isContextLost]);

  /**
   * Updates memory statistics
   */
  const updateMemoryStats = useCallback(async () => {
    const memInfo = await tf.memory();
    const gpuInfo = await tf.backend().getMemoryInfo?.();

    setMemoryStats({
      tensorCount: memInfo.numTensors,
      heapUsed: memInfo.numBytes / (1024 * 1024), // Convert to MB
      gpuMemory: gpuInfo?.numBytesInGPU || 0,
      utilizationPercentage: (memInfo.numBytes / (config.maxMemoryUsage || MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE)) * 100
    });
  }, [config.maxMemoryUsage]);

  /**
   * Cleans up resources
   */
  const cleanup = useCallback(async () => {
    if (model) {
      await model.cleanup();
      model.dispose();
    }
    if (vae) {
      vae.dispose();
    }
    tf.engine().endScope();
    tf.disposeVariables();
  }, [model, vae]);

  // Initialize model on mount
  useEffect(() => {
    initializeModel();
    return () => {
      cleanup();
    };
  }, [initializeModel, cleanup]);

  // Monitor memory usage
  useEffect(() => {
    const interval = setInterval(updateMemoryStats, 1000);
    return () => clearInterval(interval);
  }, [updateMemoryStats]);

  // Handle WebGL context loss
  useEffect(() => {
    if (isContextLost) {
      setError(new Error('WebGL context lost'));
      restoreContext();
    }
  }, [isContextLost, restoreContext]);

  return {
    isInitialized,
    isLoading,
    error,
    model,
    vae,
    generateFrame,
    trainStep,
    memoryStats,
    metrics,
    cleanup
  };
}

export default useModel;