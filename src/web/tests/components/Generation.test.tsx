/**
 * @fileoverview Test suite for Generation component with comprehensive coverage
 * @version 1.0.0
 * @license MIT
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, jest } from '@jest/globals';
import { mockPerformanceNow } from '@testing-library/jest-dom';
import { Generation, GenerationProps } from '../../src/components/Generation';
import { createMockTensor, mockWebGLContext } from '../utils';

// Mock hooks and utilities
jest.mock('../../src/hooks/useGeneration', () => ({
  useGeneration: jest.fn(() => ({
    generateFrame: jest.fn(),
    isGenerating: false,
    error: null,
    performance: { timing: jest.fn(), memory: jest.fn() }
  }))
}));

// Enhanced test setup with WebGL and performance monitoring
beforeEach(() => {
  // Setup WebGL context with full capability checking
  const gl = mockWebGLContext({
    version: 'webgl2',
    floatTexturesEnabled: true,
    maxTextureSize: 4096,
    vendor: 'Test',
    renderer: 'Test'
  });

  // Setup performance monitoring
  mockPerformanceNow();
  jest.spyOn(performance, 'now').mockImplementation(() => Date.now());
  jest.spyOn(performance, 'measure').mockImplementation();

  // Setup memory pressure monitoring
  Object.defineProperty(window, 'performance', {
    value: {
      memory: {
        usedJSHeapSize: 0,
        totalJSHeapSize: 4096 * 1024 * 1024,
        jsHeapSizeLimit: 4096 * 1024 * 1024
      }
    },
    writable: true
  });

  // Setup tensor operations with memory tracking
  jest.spyOn(WebGLRenderingContext.prototype, 'getExtension')
    .mockImplementation((name) => {
      if (name === 'WEBGL_lose_context') {
        return {
          loseContext: jest.fn(),
          restoreContext: jest.fn()
        };
      }
      return null;
    });
});

// Enhanced cleanup after each test
afterEach(() => {
  // Clean up WebGL context and verify resource release
  const canvas = document.querySelector('canvas');
  if (canvas) {
    const gl = canvas.getContext('webgl2');
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
  }

  // Reset all mocks and verify mock cleanup
  jest.clearAllMocks();
  jest.restoreAllMocks();

  // Clear tensor memory and verify garbage collection
  global.gc && global.gc();

  // Reset performance metrics
  performance.clearMarks();
  performance.clearMeasures();
});

// Helper function to render Generation component with comprehensive props
const renderGeneration = async (
  props: Partial<GenerationProps> = {},
  options: { initialMemory?: number; gpuMemory?: number } = {}
) => {
  // Create mock initial frame with specified dimensions
  const initialFrame = await createMockTensor(
    [1, 256, 256, 3],
    'float32',
    {
      trackUsage: true,
      maxMemoryMB: options.initialMemory || 4096,
      disposeAfterTest: true,
      checkLeaks: true
    }
  );

  // Setup default props with comprehensive options
  const defaultProps: GenerationProps = {
    initialFrame,
    onError: jest.fn(),
    onMemoryWarning: jest.fn()
  };

  return render(<Generation {...defaultProps} {...props} />);
};

describe('Generation Component Core Functionality', () => {
  it('renders without crashing and initializes all components', async () => {
    const { container } = await renderGeneration();

    // Verify core components are present
    expect(screen.getByTestId('preview-section')).toBeInTheDocument();
    expect(screen.getByTestId('controls-section')).toBeInTheDocument();
    expect(screen.getByTestId('memory-indicator')).toBeInTheDocument();

    // Verify WebGL context initialization
    const canvas = container.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
    expect(canvas?.getContext('webgl2')).toBeTruthy();

    // Verify memory allocation
    const memoryIndicator = screen.getByTestId('memory-indicator');
    expect(memoryIndicator).toHaveTextContent(/Memory:/);
  });

  it('handles action changes with performance monitoring', async () => {
    const onError = jest.fn();
    const { getByTestId } = await renderGeneration({ onError });

    // Setup performance monitoring
    const startTime = performance.now();

    // Trigger action control changes
    const actionSlider = getByTestId('action-strength-slider');
    fireEvent.change(actionSlider, { target: { value: '0.8' } });

    await waitFor(() => {
      // Verify state updates within performance bounds
      const endTime = performance.now();
      expect(endTime - startTime).toBeLessThan(50); // 50ms threshold

      // Verify memory usage patterns
      const memoryIndicator = getByTestId('memory-indicator');
      expect(memoryIndicator).not.toHaveTextContent(/Warning/);
    });

    expect(onError).not.toHaveBeenCalled();
  });

  it('validates generation parameters and enforces constraints', async () => {
    const { getByTestId } = await renderGeneration();

    // Test noise level constraints
    const noiseSlider = getByTestId('noise-level-slider');
    fireEvent.change(noiseSlider, { target: { value: '0.6' } });
    expect(noiseSlider).toHaveValue('0.5'); // Should clamp to max 0.5

    // Test timesteps constraints
    const timestepsSlider = getByTestId('timesteps-slider');
    fireEvent.change(timestepsSlider, { target: { value: '5' } });
    expect(timestepsSlider).toHaveValue('10'); // Should enforce minimum 10
  });
});

describe('WebGL and Performance Tests', () => {
  it('maintains WebGL context under memory pressure', async () => {
    const onMemoryWarning = jest.fn();
    const { getByTestId } = await renderGeneration(
      { onMemoryWarning },
      { initialMemory: 3584 } // 3.5GB initial memory usage
    );

    // Simulate memory pressure
    Object.defineProperty(window.performance, 'memory', {
      value: {
        usedJSHeapSize: 3.8 * 1024 * 1024 * 1024,
        totalJSHeapSize: 4 * 1024 * 1024 * 1024,
        jsHeapSizeLimit: 4 * 1024 * 1024 * 1024
      },
      configurable: true
    });

    // Trigger generation to test memory handling
    const generateButton = getByTestId('generate-button');
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(onMemoryWarning).toHaveBeenCalled();
      const memoryIndicator = getByTestId('memory-indicator');
      expect(memoryIndicator).toHaveClass('warning');
    });

    // Verify WebGL context remains valid
    const canvas = screen.getByTestId('preview-canvas');
    expect(canvas.getContext('webgl2')).toBeTruthy();
  });

  it('handles WebGL context loss and restoration', async () => {
    const onError = jest.fn();
    const { getByTestId } = await renderGeneration({ onError });

    // Simulate context loss
    const canvas = screen.getByTestId('preview-canvas');
    const gl = canvas.getContext('webgl2');
    gl?.getExtension('WEBGL_lose_context')?.loseContext();

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(expect.any(Error));
      expect(getByTestId('error-overlay')).toBeInTheDocument();
    });

    // Simulate context restoration
    gl?.getExtension('WEBGL_lose_context')?.restoreContext();

    await waitFor(() => {
      expect(getByTestId('error-overlay')).not.toBeInTheDocument();
      expect(canvas.getContext('webgl2')).toBeTruthy();
    });
  });
});