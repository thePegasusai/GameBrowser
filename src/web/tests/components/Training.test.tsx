import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import 'jest-webgl-canvas-mock';
import Training, { TrainingProps } from '../../src/components/Training';

// Mock the useTraining hook
jest.mock('../../src/hooks/useTraining', () => ({
  useTraining: jest.fn(() => ({
    trainingState: {
      isTraining: false,
      currentEpoch: 0,
      currentBatch: 0,
      loss: 0,
      progress: 0,
      status: 'idle',
      config: {
        batchSize: 4,
        learningRate: 0.001,
        epochs: 10,
        optimizerType: 'adam'
      }
    },
    memoryStats: {
      heapUsed: 1024 * 1024 * 1024, // 1GB
      heapTotal: 4 * 1024 * 1024 * 1024, // 4GB
      gpuMemory: 2 * 1024 * 1024 * 1024, // 2GB
      utilizationPercentage: 0.25
    },
    browserSupport: {
      isCompatible: true,
      webglSupported: true,
      warnings: []
    },
    startTraining: jest.fn(),
    pauseTraining: jest.fn(),
    resumeTraining: jest.fn(),
    stopTraining: jest.fn(),
    cleanupResources: jest.fn(),
    handleContextLoss: jest.fn()
  }))
}));

// Test constants
const TEST_MODEL_ID = 'test-model-123';
const MEMORY_THRESHOLD = 4096; // 4GB in MB
const BATCH_TIME_LIMIT = 200; // 200ms per batch

describe('Training component', () => {
  // Default props for testing
  const defaultProps: TrainingProps = {
    modelId: TEST_MODEL_ID,
    onMemoryPressure: jest.fn(),
    onContextLoss: jest.fn(),
    onError: jest.fn()
  };

  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
    
    // Mock WebGL context
    const mockGL = {
      createBuffer: jest.fn(),
      bindBuffer: jest.fn(),
      bufferData: jest.fn(),
      getExtension: jest.fn(() => ({
        loseContext: jest.fn(),
        restoreContext: jest.fn()
      }))
    };
    HTMLCanvasElement.prototype.getContext = jest.fn(() => mockGL);
  });

  it('renders training interface with initial state', () => {
    render(<Training {...defaultProps} />);

    // Verify core UI elements are present
    expect(screen.getByText(/Batch Size/i)).toBeInTheDocument();
    expect(screen.getByText(/Learning Rate/i)).toBeInTheDocument();
    expect(screen.getByText(/Epochs/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Start Training/i })).toBeEnabled();
  });

  it('validates training parameters against constraints', async () => {
    render(<Training {...defaultProps} />);

    // Test batch size validation
    const batchSizeInput = screen.getByLabelText(/Batch Size/i);
    await userEvent.clear(batchSizeInput);
    await userEvent.type(batchSizeInput, '100'); // Exceeds max batch size

    // Verify error message appears
    expect(screen.getByText(/Batch size must be between/i)).toBeInTheDocument();
  });

  it('monitors memory usage and triggers warnings', async () => {
    const { rerender } = render(<Training {...defaultProps} />);

    // Simulate high memory usage
    const highMemoryState = {
      heapUsed: 3.5 * 1024 * 1024 * 1024, // 3.5GB
      heapTotal: 4 * 1024 * 1024 * 1024,
      utilizationPercentage: 0.9
    };

    // Update component with high memory usage
    jest.mock('../../src/hooks/useTraining', () => ({
      useTraining: jest.fn(() => ({
        memoryStats: highMemoryState
      }))
    }));

    rerender(<Training {...defaultProps} />);

    // Verify memory warning is displayed
    expect(screen.getByText(/High memory usage detected/i)).toBeInTheDocument();
    expect(defaultProps.onMemoryPressure).toHaveBeenCalled();
  });

  it('handles WebGL context loss gracefully', async () => {
    render(<Training {...defaultProps} />);

    // Simulate WebGL context loss
    const canvas = document.querySelector('canvas');
    const contextLossEvent = new Event('webglcontextlost');
    fireEvent(canvas!, contextLossEvent);

    // Verify error handling
    await waitFor(() => {
      expect(screen.getByText(/WebGL context lost/i)).toBeInTheDocument();
      expect(defaultProps.onContextLoss).toHaveBeenCalled();
    });
  });

  it('manages training state transitions correctly', async () => {
    const { rerender } = render(<Training {...defaultProps} />);

    // Start training
    const startButton = screen.getByRole('button', { name: /Start Training/i });
    await userEvent.click(startButton);

    // Verify training started
    expect(screen.getByRole('button', { name: /Pause/i })).toBeInTheDocument();

    // Simulate training progress
    const trainingState = {
      isTraining: true,
      currentEpoch: 1,
      currentBatch: 10,
      loss: 0.5,
      progress: 0.25,
      status: 'training'
    };

    rerender(<Training {...defaultProps} />);

    // Verify progress updates
    expect(screen.getByText(/Epoch: 1/i)).toBeInTheDocument();
    expect(screen.getByText(/Loss: 0.5/i)).toBeInTheDocument();
  });

  it('validates training performance against time constraints', async () => {
    render(<Training {...defaultProps} />);

    // Simulate slow training step
    const slowTrainingState = {
      stepTime: BATCH_TIME_LIMIT + 50, // Exceeds 200ms limit
      status: 'training'
    };

    // Update component with slow training state
    jest.mock('../../src/hooks/useTraining', () => ({
      useTraining: jest.fn(() => ({
        trainingState: slowTrainingState
      }))
    }));

    // Verify performance warning
    expect(screen.getByText(/exceeded time threshold/i)).toBeInTheDocument();
  });

  it('cleans up resources on unmount', () => {
    const { unmount } = render(<Training {...defaultProps} />);
    
    // Get reference to cleanup function
    const { cleanupResources } = require('../../src/hooks/useTraining').useTraining();
    
    // Unmount component
    unmount();

    // Verify cleanup was called
    expect(cleanupResources).toHaveBeenCalled();
  });
});