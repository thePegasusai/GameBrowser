/**
 * Test suite for VideoUpload component with memory management and WebGL acceleration
 * @version 1.0.0
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { jest } from '@jest/globals';
import VideoUpload from '../../src/components/VideoUpload';
import { setupGlobalMocks } from '../setup';
import { createMockVideoFrame } from '../utils';
import { VideoProcessingState } from '../../src/types/video';
import { MEMORY_CONSTRAINTS } from '../../src/constants/model';

// Mock performance API
const mockPerformanceAPI = {
  memory: {
    usedJSHeapSize: 0,
    totalJSHeapSize: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024,
    jsHeapSizeLimit: 4096 * 1024 * 1024
  }
};

// Mock callbacks
const mockOnUploadComplete = jest.fn();
const mockOnError = jest.fn();

// Default test props
const DEFAULT_PROPS = {
  maxFileSize: 100000000, // 100MB
  acceptedFormats: ['video/mp4', 'video/webm'],
  onUploadComplete: mockOnUploadComplete,
  onError: mockOnError,
  maxMemoryUsage: 4096, // 4GB
  frameProcessingTimeout: 50 // 50ms per frame
};

describe('VideoUpload Component', () => {
  // Setup before each test
  beforeEach(() => {
    setupGlobalMocks();
    jest.clearAllMocks();
    // Mock performance API
    Object.defineProperty(window, 'performance', {
      value: { ...performance, ...mockPerformanceAPI }
    });
  });

  // Cleanup after each test
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Component Rendering', () => {
    it('renders upload area with correct instructions', () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      expect(screen.getByText(/drag and drop video/i)).toBeInTheDocument();
      expect(screen.getByText(/supported formats/i)).toBeInTheDocument();
    });

    it('displays memory usage indicator', () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const memoryIndicator = screen.getByText(/memory:/i);
      expect(memoryIndicator).toBeInTheDocument();
      expect(memoryIndicator).toHaveTextContent(/0mb/i);
    });
  });

  describe('File Upload Handling', () => {
    it('accepts valid video files', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const file = new File([''], 'test.mp4', { type: 'video/mp4' });
      const input = screen.getByRole('button');
      
      fireEvent.drop(input, {
        dataTransfer: {
          files: [file]
        }
      });

      await waitFor(() => {
        expect(screen.getByText(/processing video/i)).toBeInTheDocument();
      });
    });

    it('rejects files exceeding size limit', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const largeFile = new File([''], 'large.mp4', {
        type: 'video/mp4'
      });
      Object.defineProperty(largeFile, 'size', { value: DEFAULT_PROPS.maxFileSize + 1 });
      
      const input = screen.getByRole('button');
      fireEvent.drop(input, {
        dataTransfer: {
          files: [largeFile]
        }
      });

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(expect.any(Error));
        expect(screen.getByText(/file size exceeds/i)).toBeInTheDocument();
      });
    });

    it('validates file format', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const invalidFile = new File([''], 'test.txt', { type: 'text/plain' });
      const input = screen.getByRole('button');
      
      fireEvent.drop(input, {
        dataTransfer: {
          files: [invalidFile]
        }
      });

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(expect.any(Error));
        expect(screen.getByText(/unsupported file type/i)).toBeInTheDocument();
      });
    });
  });

  describe('Memory Management', () => {
    it('monitors memory usage during processing', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const file = new File([''], 'test.mp4', { type: 'video/mp4' });
      const input = screen.getByRole('button');
      
      // Mock memory pressure
      Object.defineProperty(window.performance, 'memory', {
        value: {
          ...mockPerformanceAPI.memory,
          usedJSHeapSize: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.9 * 1024 * 1024
        }
      });

      fireEvent.drop(input, {
        dataTransfer: {
          files: [file]
        }
      });

      await waitFor(() => {
        const memoryIndicator = screen.getByText(/memory:/i);
        expect(memoryIndicator).toHaveTextContent(/high usage/i);
      });
    });

    it('triggers cleanup on memory pressure', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      // Simulate memory pressure event
      const memoryPressureEvent = new Event('memorypressure');
      window.dispatchEvent(memoryPressureEvent);

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(
          expect.objectContaining({
            message: expect.stringContaining('memory')
          })
        );
      });
    });
  });

  describe('WebGL Context', () => {
    it('handles WebGL context loss gracefully', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const canvas = document.createElement('canvas');
      const contextLossEvent = new Event('webglcontextlost');
      canvas.dispatchEvent(contextLossEvent);

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(
          expect.objectContaining({
            message: expect.stringContaining('WebGL context lost')
          })
        );
      });
    });

    it('validates WebGL capabilities on mount', () => {
      const { container } = render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const canvas = container.querySelector('canvas');
      expect(canvas?.getContext('webgl2')).toBeTruthy();
    });
  });

  describe('Processing State', () => {
    it('shows progress during video processing', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const file = new File([''], 'test.mp4', { type: 'video/mp4' });
      const input = screen.getByRole('button');
      
      fireEvent.drop(input, {
        dataTransfer: {
          files: [file]
        }
      });

      await waitFor(() => {
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
      });
    });

    it('handles processing completion', async () => {
      render(<VideoUpload {...DEFAULT_PROPS} />);
      
      const file = new File([''], 'test.mp4', { type: 'video/mp4' });
      const mockFrames = [createMockVideoFrame(256, 256)];
      
      const input = screen.getByRole('button');
      fireEvent.drop(input, {
        dataTransfer: {
          files: [file]
        }
      });

      await waitFor(() => {
        expect(mockOnUploadComplete).toHaveBeenCalledWith(
          expect.arrayContaining([
            expect.objectContaining({
              width: 256,
              height: 256
            })
          ])
        );
      });
    });
  });
});