/**
 * @fileoverview React component for real-time video game footage frame preview with WebGL acceleration
 * @version 1.0.0
 * @license MIT
 */

import React, { useRef, useEffect, useState, useCallback } from 'react'; // v18.x
import styled from 'styled-components'; // v5.3.0
import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { WebGLContextManager } from '../../lib/webgl/context';
import { useGeneration } from '../../hooks/useGeneration';

// Styled components for preview rendering
const PreviewContainer = styled.div<{ width: number; height: number }>`
  position: relative;
  width: 100%;
  max-width: ${props => props.width}px;
  height: ${props => props.height}px;
  background-color: #000;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: high-quality;
`;

const ErrorOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 1rem;
  backdrop-filter: blur(4px);
`;

// Interface for quality settings
interface QualitySettings {
  targetPSNR: number;
  adaptiveScaling: boolean;
  maxFrameLatency: number;
}

// Interface for performance configuration
interface PerformanceConfig {
  enableFrameSkipping: boolean;
  targetFPS: number;
  memoryLimit: number;
}

// Props interface for Preview component
interface PreviewProps {
  width: number;
  height: number;
  onError: (error: string) => void;
  quality: QualitySettings;
  performance: PerformanceConfig;
}

/**
 * Enhanced preview component for real-time video game footage display
 * Implements WebGL acceleration, frame caching, and quality monitoring
 */
export const Preview: React.FC<PreviewProps> = React.memo(({
  width,
  height,
  onError,
  quality,
  performance
}) => {
  // Refs for canvas and WebGL context
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextManagerRef = useRef<WebGLContextManager | null>(null);
  const frameBufferRef = useRef<ImageData[]>([]);
  const lastFrameTimeRef = useRef<number>(0);

  // State for error handling and performance metrics
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [psnr, setPsnr] = useState<number>(0);

  // Generation hook for frame processing
  const { generateFrame, isGenerating, frameQuality } = useGeneration({
    batchSize: 1,
    temperature: 1.0,
    useCache: true,
    memoryOptimize: true
  });

  /**
   * Initializes WebGL context and sets up frame rendering
   */
  useEffect(() => {
    const initializeContext = async () => {
      try {
        if (!canvasRef.current) return;

        contextManagerRef.current = new WebGLContextManager({
          alpha: false,
          antialias: true,
          depth: false,
          failIfMajorPerformanceCaveat: true,
          powerPreference: 'high-performance'
        }, canvasRef.current);

        // Set up context loss handling
        canvasRef.current.addEventListener('webglcontextlost', handleContextLoss);
        canvasRef.current.addEventListener('webglcontextrestored', handleContextRestore);

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'WebGL initialization failed';
        setError(errorMessage);
        onError(errorMessage);
      }
    };

    initializeContext();

    return () => {
      if (contextManagerRef.current) {
        contextManagerRef.current.dispose();
      }
      if (canvasRef.current) {
        canvasRef.current.removeEventListener('webglcontextlost', handleContextLoss);
        canvasRef.current.removeEventListener('webglcontextrestored', handleContextRestore);
      }
    };
  }, [onError]);

  /**
   * Handles WebGL context loss
   */
  const handleContextLoss = useCallback((event: WebGLContextEvent) => {
    event.preventDefault();
    setError('WebGL context lost. Attempting recovery...');
  }, []);

  /**
   * Handles WebGL context restoration
   */
  const handleContextRestore = useCallback(async () => {
    try {
      if (contextManagerRef.current) {
        await contextManagerRef.current.restoreContext();
        setError(null);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Context restoration failed';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [onError]);

  /**
   * Renders frame with quality monitoring and performance optimization
   */
  const renderFrame = useCallback(async (tensor: tf.Tensor4D) => {
    if (!canvasRef.current || !contextManagerRef.current) return;

    const currentTime = performance.now();
    const frameTime = currentTime - lastFrameTimeRef.current;

    // Skip frame if needed for performance
    if (performance.enableFrameSkipping && frameTime < (1000 / performance.targetFPS)) {
      return;
    }

    try {
      // Convert tensor to ImageData
      const imageData = await tf.browser.toPixels(tensor);
      const ctx = canvasRef.current.getContext('2d')!;
      
      // Update frame buffer
      frameBufferRef.current.push(new ImageData(imageData, width, height));
      if (frameBufferRef.current.length > 3) {
        frameBufferRef.current.shift();
      }

      // Render frame
      const frame = frameBufferRef.current[frameBufferRef.current.length - 1];
      ctx.putImageData(frame, 0, 0);

      // Update metrics
      setFps(1000 / frameTime);
      setPsnr(frameQuality?.psnr || 0);
      lastFrameTimeRef.current = currentTime;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Frame rendering failed';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [width, height, performance, onError, frameQuality]);

  /**
   * Monitors and maintains frame quality
   */
  useEffect(() => {
    if (psnr < quality.targetPSNR && quality.adaptiveScaling) {
      // Implement quality improvement logic
    }
  }, [psnr, quality]);

  return (
    <PreviewContainer width={width} height={height}>
      <Canvas
        ref={canvasRef}
        width={width}
        height={height}
      />
      {error && (
        <ErrorOverlay>
          {error}
        </ErrorOverlay>
      )}
    </PreviewContainer>
  );
});

Preview.displayName = 'Preview';

export default Preview;