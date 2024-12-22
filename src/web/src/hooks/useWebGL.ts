/**
 * @fileoverview React hook for managing WebGL context and GPU acceleration
 * @version 1.0.0
 * @license MIT
 */

import { useEffect, useCallback, useRef } from 'react'; // v18.0.0
import { useErrorBoundary } from 'react'; // v18.0.0
import { WebGLContextManager } from '../lib/webgl/context';
import { WebGLBufferManager } from '../lib/webgl/buffers';
import { DEFAULT_WEBGL_CONFIG } from '../config/webgl';

/**
 * Interface for WebGL performance metrics
 */
interface WebGLMetrics {
  frameTime: number;
  memoryUsage: number;
  gpuUtilization: number;
  isPerformant: boolean;
  lastUpdate: number;
}

/**
 * Interface for WebGL capabilities
 */
interface WebGLCapabilities {
  maxTextureSize: number;
  vendor: string;
  renderer: string;
  version: string;
  extensions: string[];
}

/**
 * Enhanced React hook for WebGL context management
 */
export function useWebGL(config = DEFAULT_WEBGL_CONFIG) {
  // Refs for persistent instances
  const contextManagerRef = useRef<WebGLContextManager | null>(null);
  const bufferManagerRef = useRef<WebGLBufferManager | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const metricsRef = useRef<WebGLMetrics>({
    frameTime: 0,
    memoryUsage: 0,
    gpuUtilization: 0,
    isPerformant: true,
    lastUpdate: Date.now()
  });

  // Error boundary integration
  const { showBoundary } = useErrorBoundary();

  /**
   * Initializes WebGL context with error handling
   */
  const initializeContext = useCallback(async () => {
    try {
      if (!contextManagerRef.current) {
        contextManagerRef.current = new WebGLContextManager(config);
        
        // Initialize buffer manager after context
        bufferManagerRef.current = new WebGLBufferManager(
          contextManagerRef.current,
          {
            maxPoolSize: 100,
            reuseThreshold: 1000,
            cleanupInterval: 30000
          }
        );

        // Store canvas reference
        canvasRef.current = contextManagerRef.current.getCanvas();
      }
      return true;
    } catch (error) {
      showBoundary(error);
      return false;
    }
  }, [config, showBoundary]);

  /**
   * Checks WebGL capabilities and features
   */
  const checkCapabilities = useCallback((): WebGLCapabilities | null => {
    const gl = contextManagerRef.current?.getContext();
    if (!gl) return null;

    try {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      return {
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        version: gl.getParameter(gl.VERSION),
        extensions: gl.getSupportedExtensions() || []
      };
    } catch (error) {
      showBoundary(error);
      return null;
    }
  }, [showBoundary]);

  /**
   * Monitors WebGL performance metrics
   */
  const monitorPerformance = useCallback(() => {
    if (!contextManagerRef.current) return;

    const now = Date.now();
    const metrics = metricsRef.current;
    
    try {
      // Update metrics every 1000ms
      if (now - metrics.lastUpdate >= 1000) {
        const gl = contextManagerRef.current.getContext();
        const ext = gl?.getExtension('WEBGL_debug_renderer_info');
        
        if (gl && ext) {
          // Calculate frame time
          const frameStart = performance.now();
          gl.finish();
          metrics.frameTime = performance.now() - frameStart;

          // Estimate GPU utilization
          metrics.gpuUtilization = metrics.frameTime > 16 ? 100 : (metrics.frameTime / 16) * 100;

          // Check memory usage
          const memoryInfo = (performance as any).memory;
          if (memoryInfo) {
            metrics.memoryUsage = memoryInfo.totalJSHeapSize / memoryInfo.jsHeapSizeLimit;
          }

          // Update performance status
          metrics.isPerformant = metrics.frameTime < 50 && metrics.memoryUsage < 0.9;
          metrics.lastUpdate = now;
        }
      }
    } catch (error) {
      console.error('Performance monitoring error:', error);
    }
  }, []);

  /**
   * Creates a WebGL buffer with memory management
   */
  const createBuffer = useCallback((data: ArrayBuffer, target?: number, usage?: number) => {
    try {
      return bufferManagerRef.current?.createBuffer(data, target, usage);
    } catch (error) {
      showBoundary(error);
      return null;
    }
  }, [showBoundary]);

  /**
   * Deletes a WebGL buffer
   */
  const deleteBuffer = useCallback((bufferId: string) => {
    try {
      bufferManagerRef.current?.deleteBuffer(bufferId);
    } catch (error) {
      showBoundary(error);
    }
  }, [showBoundary]);

  /**
   * Reinitializes context after loss
   */
  const reinitializeContext = useCallback(async () => {
    try {
      await contextManagerRef.current?.dispose();
      return initializeContext();
    } catch (error) {
      showBoundary(error);
      return false;
    }
  }, [initializeContext, showBoundary]);

  // Initialize context on mount
  useEffect(() => {
    initializeContext();

    // Cleanup on unmount
    return () => {
      try {
        contextManagerRef.current?.dispose();
        contextManagerRef.current = null;
        bufferManagerRef.current = null;
      } catch (error) {
        console.error('Cleanup error:', error);
      }
    };
  }, [initializeContext]);

  // Set up performance monitoring
  useEffect(() => {
    const monitorInterval = setInterval(monitorPerformance, 1000);
    return () => clearInterval(monitorInterval);
  }, [monitorPerformance]);

  return {
    gl: contextManagerRef.current?.getContext() || null,
    canvas: canvasRef.current,
    isContextLost: contextManagerRef.current?.getContext()?.isContextLost() || false,
    createBuffer,
    deleteBuffer,
    reinitializeContext,
    checkCapabilities,
    monitorPerformance,
    metrics: metricsRef.current
  };
}

export default useWebGL;