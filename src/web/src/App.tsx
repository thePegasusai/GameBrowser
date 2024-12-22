/**
 * Root application component for browser-based video game diffusion model
 * Implements memory-aware processing and WebGL context management
 * @version 1.0.0
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Alert } from '@mui/material'; // v5.0.0
import Layout from './components/Layout';
import VideoUpload from './components/VideoUpload';
import { validateMemoryConstraints } from './lib/utils/validation';
import { Logger } from './lib/utils/logger';
import { VideoFrame } from './types/video';
import { MEMORY_CONSTRAINTS, PERFORMANCE_THRESHOLDS } from './constants/model';

// Initialize logger
const logger = new Logger({
  level: 'info',
  namespace: 'app',
  persistLogs: true,
  metricsRetentionMs: 3600000
});

// Application state interfaces
interface AppState {
  currentStep: VideoStep;
  modelId: string | null;
  initialFrame: tf.Tensor4D | null;
  error: ErrorDetails | null;
  memoryUsage: MemoryStats;
  webglContext: WebGLContextStatus;
}

interface ErrorDetails {
  message: string;
  code: string;
  severity: 'error' | 'warning' | 'info';
  recoverable: boolean;
}

interface MemoryStats {
  heapUsed: number;
  tensorCount: number;
  webglMemory: number;
}

type VideoStep = 'upload' | 'process' | 'generate' | 'complete';

/**
 * Root application component with memory management and error handling
 */
const App: React.FC = () => {
  // Application state
  const [state, setState] = useState<AppState>({
    currentStep: 'upload',
    modelId: null,
    initialFrame: null,
    error: null,
    memoryUsage: {
      heapUsed: 0,
      tensorCount: 0,
      webglMemory: 0
    },
    webglContext: {
      isAvailable: false,
      version: null,
      maxTextureSize: 0
    }
  });

  // Initialize WebGL context and validate browser capabilities
  useEffect(() => {
    const initializeContext = async () => {
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');

        if (!gl) {
          throw new Error('WebGL 2.0 is required but not available');
        }

        // Validate WebGL capabilities
        const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        if (maxTextureSize < 2048) {
          throw new Error('Insufficient WebGL texture support');
        }

        setState(prev => ({
          ...prev,
          webglContext: {
            isAvailable: true,
            version: gl.getParameter(gl.VERSION),
            maxTextureSize
          }
        }));

        logger.info('WebGL context initialized', {
          version: gl.getParameter(gl.VERSION),
          maxTextureSize,
          vendor: gl.getParameter(gl.VENDOR)
        });

      } catch (error) {
        handleError({
          message: error instanceof Error ? error.message : 'WebGL initialization failed',
          code: 'WEBGL_INIT_ERROR',
          severity: 'error',
          recoverable: false
        });
      }
    };

    initializeContext();
  }, []);

  // Monitor memory usage
  useEffect(() => {
    const checkMemory = async () => {
      try {
        const memoryResult = await validateMemoryConstraints(
          MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
          { gpuMemory: MEMORY_CONSTRAINTS.MAX_TENSOR_BUFFER_SIZE }
        );

        setState(prev => ({
          ...prev,
          memoryUsage: {
            heapUsed: memoryResult.availableMemory,
            tensorCount: performance.memory?.usedJSHeapSize || 0,
            webglMemory: memoryResult.requiredMemory
          }
        }));

        if (!memoryResult.canAllocate) {
          handleMemoryPressure(memoryResult.availableMemory);
        }

      } catch (error) {
        logger.error('Memory check failed', error);
      }
    };

    const intervalId = setInterval(checkMemory, MEMORY_CONSTRAINTS.MEMORY_CHECK_INTERVAL);
    return () => clearInterval(intervalId);
  }, []);

  // Handle video upload completion
  const handleUploadComplete = useCallback((frames: VideoFrame[]) => {
    try {
      if (frames.length === 0) {
        throw new Error('No frames extracted from video');
      }

      setState(prev => ({
        ...prev,
        currentStep: 'process',
        initialFrame: frames[0].data as tf.Tensor4D
      }));

      logger.info('Video upload complete', {
        frameCount: frames.length,
        firstFrameShape: frames[0].data.shape
      });

    } catch (error) {
      handleError({
        message: error instanceof Error ? error.message : 'Video processing failed',
        code: 'UPLOAD_ERROR',
        severity: 'error',
        recoverable: true
      });
    }
  }, []);

  // Handle memory pressure situations
  const handleMemoryPressure = useCallback((currentUsage: number) => {
    logger.warn('Memory pressure detected', { currentUsage });
    
    setState(prev => ({
      ...prev,
      error: {
        message: 'High memory usage detected. Consider reducing video size.',
        code: 'MEMORY_PRESSURE',
        severity: 'warning',
        recoverable: true
      }
    }));
  }, []);

  // Handle errors
  const handleError = useCallback((error: ErrorDetails) => {
    setState(prev => ({
      ...prev,
      error,
      currentStep: error.recoverable ? prev.currentStep : 'upload'
    }));

    logger.error('Application error', error);
  }, []);

  // Clear error state
  const handleErrorClose = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  return (
    <Layout version={process.env.REACT_APP_VERSION || '1.0.0'}>
      {state.error && (
        <Alert 
          severity={state.error.severity}
          onClose={state.error.recoverable ? handleErrorClose : undefined}
          sx={{ mb: 2 }}
        >
          {state.error.message}
        </Alert>
      )}

      {!state.webglContext.isAvailable ? (
        <Alert severity="error">
          WebGL 2.0 is required for this application.
        </Alert>
      ) : (
        <VideoUpload
          onUploadComplete={handleUploadComplete}
          onError={handleError}
          onMemoryPressure={handleMemoryPressure}
        />
      )}
    </Layout>
  );
};

export default App;