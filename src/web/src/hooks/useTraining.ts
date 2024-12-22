/**
 * @fileoverview React hook for managing training process of browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { useState, useEffect, useCallback, useRef } from 'react'; // v18.x
import * as tf from '@tensorflow/tfjs'; // v4.x
import { wrap } from 'comlink'; // v4.x
import { TrainingWorker } from '../workers/training.worker';
import { getTrainingConfig } from '../config/training';
import { ModelConfig } from '../types/model';

/**
 * Training state interface with enhanced monitoring capabilities
 */
interface TrainingState {
  isTraining: boolean;
  currentEpoch: number;
  currentBatch: number;
  loss: number;
  progress: number;
  status: string;
  memoryUsage: number;
  stepTime: number;
  webglSupported: boolean;
  browserCompatibility: string;
}

/**
 * Memory monitoring interface
 */
interface MemoryStats {
  heapUsed: number;
  gpuMemory: number;
  tensorCount: number;
  utilizationPercentage: number;
}

/**
 * Browser support information
 */
interface BrowserSupport {
  isCompatible: boolean;
  webglVersion: string;
  warnings: string[];
}

// Constants for memory and performance thresholds
const MEMORY_THRESHOLD = 3.5 * 1024 * 1024 * 1024; // 3.5GB
const PERFORMANCE_CHECK_INTERVAL = 1000; // 1 second
const CLEANUP_INTERVAL = 5000; // 5 seconds

/**
 * Custom hook for managing the training process with enhanced monitoring
 */
export function useTraining(modelConfig: ModelConfig) {
  // Training state management
  const [trainingState, setTrainingState] = useState<TrainingState>({
    isTraining: false,
    currentEpoch: 0,
    currentBatch: 0,
    loss: 0,
    progress: 0,
    status: 'idle',
    memoryUsage: 0,
    stepTime: 0,
    webglSupported: false,
    browserCompatibility: 'checking'
  });

  // Memory monitoring
  const [memoryStats, setMemoryStats] = useState<MemoryStats>({
    heapUsed: 0,
    gpuMemory: 0,
    tensorCount: 0,
    utilizationPercentage: 0
  });

  // Browser compatibility tracking
  const [browserSupport, setBrowserSupport] = useState<BrowserSupport>({
    isCompatible: false,
    webglVersion: '',
    warnings: []
  });

  // Worker and interval references
  const workerRef = useRef<Worker | null>(null);
  const trainingWorkerRef = useRef<any>(null);
  const memoryCheckInterval = useRef<number>();
  const performanceCheckInterval = useRef<number>();

  /**
   * Initialize training environment and check compatibility
   */
  useEffect(() => {
    const initializeTraining = async () => {
      try {
        // Check WebGL support
        const webglSupported = tf.backend() === 'webgl';
        const webglVersion = await getWebGLVersion();

        // Validate browser compatibility
        const compatibility = await validateBrowserSupport();
        setBrowserSupport({
          isCompatible: compatibility.isCompatible,
          webglVersion,
          warnings: compatibility.warnings
        });

        // Initialize worker if compatible
        if (compatibility.isCompatible) {
          initializeWorker();
        }

        setTrainingState(prev => ({
          ...prev,
          webglSupported,
          browserCompatibility: compatibility.isCompatible ? 'compatible' : 'incompatible'
        }));
      } catch (error) {
        console.error('Training initialization error:', error);
        setTrainingState(prev => ({
          ...prev,
          status: 'error',
          browserCompatibility: 'error'
        }));
      }
    };

    initializeTraining();

    return () => cleanup();
  }, [modelConfig]);

  /**
   * Initialize and configure training worker
   */
  const initializeWorker = useCallback(async () => {
    try {
      // Create and wrap worker with Comlink
      const worker = new Worker(new URL('../workers/training.worker', import.meta.url));
      const wrappedWorker = wrap<TrainingWorker>(worker);
      
      workerRef.current = worker;
      trainingWorkerRef.current = wrappedWorker;

      // Get optimized training configuration
      const deviceMemory = await getDeviceMemoryInfo();
      const trainingConfig = getTrainingConfig(modelConfig, deviceMemory);

      // Initialize worker with configuration
      await wrappedWorker.initialize({
        modelConfig,
        trainingConfig,
        webglSupported: browserSupport.isCompatible
      });

      // Set up message handlers
      worker.onmessage = handleWorkerMessage;
      worker.onerror = handleWorkerError;

    } catch (error) {
      console.error('Worker initialization error:', error);
      setTrainingState(prev => ({ ...prev, status: 'error' }));
    }
  }, [modelConfig, browserSupport.isCompatible]);

  /**
   * Start training process
   */
  const startTraining = useCallback(async () => {
    if (!trainingWorkerRef.current || !browserSupport.isCompatible) {
      return;
    }

    try {
      setTrainingState(prev => ({ ...prev, isTraining: true, status: 'training' }));
      
      // Start memory and performance monitoring
      startMonitoring();

      // Begin training in worker
      await trainingWorkerRef.current.startTraining();

    } catch (error) {
      console.error('Training start error:', error);
      setTrainingState(prev => ({ ...prev, isTraining: false, status: 'error' }));
    }
  }, [browserSupport.isCompatible]);

  /**
   * Pause training process
   */
  const pauseTraining = useCallback(async () => {
    if (!trainingWorkerRef.current) return;

    try {
      await trainingWorkerRef.current.pauseTraining();
      setTrainingState(prev => ({ ...prev, status: 'paused' }));
    } catch (error) {
      console.error('Training pause error:', error);
    }
  }, []);

  /**
   * Resume training process
   */
  const resumeTraining = useCallback(async () => {
    if (!trainingWorkerRef.current) return;

    try {
      await trainingWorkerRef.current.resumeTraining();
      setTrainingState(prev => ({ ...prev, status: 'training' }));
    } catch (error) {
      console.error('Training resume error:', error);
    }
  }, []);

  /**
   * Start memory and performance monitoring
   */
  const startMonitoring = useCallback(() => {
    // Monitor memory usage
    memoryCheckInterval.current = window.setInterval(async () => {
      const memoryInfo = await getMemoryInfo();
      setMemoryStats(memoryInfo);

      // Trigger cleanup if memory threshold exceeded
      if (memoryInfo.utilizationPercentage > 0.9) {
        await cleanupMemory();
      }
    }, CLEANUP_INTERVAL);

    // Monitor performance
    performanceCheckInterval.current = window.setInterval(() => {
      if (trainingWorkerRef.current) {
        trainingWorkerRef.current.getPerformanceMetrics();
      }
    }, PERFORMANCE_CHECK_INTERVAL);
  }, []);

  /**
   * Handle worker messages
   */
  const handleWorkerMessage = useCallback((event: MessageEvent) => {
    const { type, data } = event.data;

    switch (type) {
      case 'progress':
        setTrainingState(prev => ({
          ...prev,
          currentBatch: data.batch,
          loss: data.loss,
          progress: data.progress,
          stepTime: data.timePerStep
        }));
        break;

      case 'epoch':
        setTrainingState(prev => ({
          ...prev,
          currentEpoch: data.epoch
        }));
        break;

      case 'memoryStatus':
        setMemoryStats(data);
        break;

      case 'error':
        console.error('Worker error:', data);
        setTrainingState(prev => ({ ...prev, status: 'error' }));
        break;
    }
  }, []);

  /**
   * Handle worker errors
   */
  const handleWorkerError = useCallback((error: ErrorEvent) => {
    console.error('Worker error:', error);
    setTrainingState(prev => ({ ...prev, status: 'error' }));
  }, []);

  /**
   * Clean up resources and tensors
   */
  const cleanupMemory = useCallback(async () => {
    if (!trainingWorkerRef.current) return;

    try {
      await trainingWorkerRef.current.cleanupTensors();
      tf.engine().startScope();
      tf.engine().endScope();
    } catch (error) {
      console.error('Memory cleanup error:', error);
    }
  }, []);

  /**
   * Clean up resources on unmount
   */
  const cleanup = useCallback(() => {
    if (memoryCheckInterval.current) {
      clearInterval(memoryCheckInterval.current);
    }
    if (performanceCheckInterval.current) {
      clearInterval(performanceCheckInterval.current);
    }
    if (workerRef.current) {
      workerRef.current.terminate();
    }
    cleanupMemory();
  }, []);

  return {
    trainingState,
    memoryStats,
    browserSupport,
    startTraining,
    pauseTraining,
    resumeTraining
  };
}

// Utility functions
async function getWebGLVersion(): Promise<string> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  return gl ? gl.getParameter(gl.VERSION) : 'not supported';
}

async function validateBrowserSupport(): Promise<{ isCompatible: boolean; warnings: string[] }> {
  const warnings: string[] = [];
  
  // Check WebGL support
  if (!tf.backend() === 'webgl') {
    warnings.push('WebGL not supported');
  }

  // Check Web Workers support
  if (!window.Worker) {
    warnings.push('Web Workers not supported');
  }

  // Check memory constraints
  const memory = performance?.memory;
  if (memory && memory.jsHeapSizeLimit < MEMORY_THRESHOLD) {
    warnings.push('Insufficient memory available');
  }

  return {
    isCompatible: warnings.length === 0,
    warnings
  };
}

async function getMemoryInfo(): Promise<MemoryStats> {
  const memory = tf.memory();
  const gpuMemory = tf.backend() === 'webgl' ? 
    (tf.backend() as any).numBytesInGPU : 0;

  return {
    heapUsed: memory.numBytes,
    gpuMemory,
    tensorCount: memory.numTensors,
    utilizationPercentage: memory.numBytes / MEMORY_THRESHOLD
  };
}

async function getDeviceMemoryInfo() {
  return {
    totalGPUMemory: 4 * 1024 * 1024 * 1024, // 4GB default
    availableGPUMemory: 3 * 1024 * 1024 * 1024, // 3GB default
    totalRAM: performance?.memory?.jsHeapSizeLimit || 4 * 1024 * 1024 * 1024,
    browserMemoryLimit: MEMORY_THRESHOLD
  };
}

export default useTraining;