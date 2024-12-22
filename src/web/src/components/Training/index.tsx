/**
 * @fileoverview Main Training component for browser-based video game diffusion model
 * Orchestrates training interface with enhanced memory management and error handling
 * @version 1.0.0
 */

import React, { useCallback, useEffect, useState } from 'react';
import { styled } from '@mui/material/styles';
import { Box, Alert } from '@mui/material';
import { useMediaQuery } from '@mui/material';
import TrainingControls from './Controls';
import TrainingParameters from './Parameters';
import TrainingProgress from './Progress';
import { useTraining } from '../../hooks/useTraining';

// Styled container for the training interface
const TrainingContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: '24px',
  padding: '24px',
  width: '100%',
  maxWidth: '1200px',
  margin: '0 auto',
  position: 'relative',
  minHeight: '400px',
  overflow: 'hidden',
  [theme.breakpoints.down('md')]: {
    padding: '16px',
    gap: '16px',
  }
}));

// Interface for Training component props
interface TrainingProps {
  modelId: string;
  className?: string;
  onMemoryPressure?: () => void;
  onContextLoss?: () => void;
  onError?: (error: Error) => void;
}

/**
 * Main Training component that orchestrates the training interface
 * Manages training state, memory, and WebGL context with error handling
 */
const Training: React.FC<TrainingProps> = React.memo(({
  modelId,
  className,
  onMemoryPressure,
  onContextLoss,
  onError
}) => {
  // Get training hook state and functions
  const {
    trainingState,
    memoryStats,
    browserSupport,
    startTraining,
    cleanupResources,
    handleContextLoss
  } = useTraining();

  // Local state for error handling
  const [error, setError] = useState<string | null>(null);
  const [isMemoryWarning, setIsMemoryWarning] = useState(false);

  // Media query for responsive layout
  const isMobile = useMediaQuery('(max-width:600px)');

  /**
   * Handle training configuration changes with validation
   */
  const handleConfigChange = useCallback(async (config) => {
    try {
      // Reset previous errors
      setError(null);
      setIsMemoryWarning(false);

      // Check memory requirements
      const requiredMemory = config.batchSize * 1024 * 1024; // Estimate memory per batch
      if (memoryStats.heapUsed + requiredMemory > memoryStats.heapTotal * 0.9) {
        setIsMemoryWarning(true);
        onMemoryPressure?.();
      }

      // Update training configuration
      await startTraining(config);
    } catch (err) {
      setError(err.message);
      onError?.(err);
    }
  }, [memoryStats, onMemoryPressure, onError, startTraining]);

  /**
   * Monitor WebGL context and memory pressure
   */
  useEffect(() => {
    const cleanup = () => {
      cleanupResources();
    };

    // Handle WebGL context loss
    const handleContextLossEvent = () => {
      handleContextLoss();
      onContextLoss?.();
      setError('WebGL context lost. Training paused.');
    };

    // Add context loss listener
    const canvas = document.querySelector('canvas');
    if (canvas) {
      canvas.addEventListener('webglcontextlost', handleContextLossEvent);
    }

    // Monitor memory usage
    if (memoryStats.utilizationPercentage > 0.9) {
      setIsMemoryWarning(true);
      onMemoryPressure?.();
    }

    return () => {
      cleanup();
      if (canvas) {
        canvas.removeEventListener('webglcontextlost', handleContextLossEvent);
      }
    };
  }, [
    cleanupResources,
    handleContextLoss,
    memoryStats.utilizationPercentage,
    onContextLoss,
    onMemoryPressure
  ]);

  return (
    <TrainingContainer className={className}>
      {/* Error display */}
      {error && (
        <Alert 
          severity="error" 
          onClose={() => setError(null)}
          sx={{ mb: 2 }}
        >
          {error}
        </Alert>
      )}

      {/* Memory warning */}
      {isMemoryWarning && (
        <Alert 
          severity="warning"
          onClose={() => setIsMemoryWarning(false)}
          sx={{ mb: 2 }}
        >
          High memory usage detected. Consider reducing batch size.
        </Alert>
      )}

      {/* Browser compatibility warning */}
      {!browserSupport.isCompatible && (
        <Alert 
          severity="warning"
          sx={{ mb: 2 }}
        >
          {`Browser compatibility issues detected: ${browserSupport.warnings.join(', ')}`}
        </Alert>
      )}

      {/* Training parameters configuration */}
      <TrainingParameters
        onConfigChange={handleConfigChange}
        disabled={trainingState.isTraining}
      />

      {/* Training controls */}
      <TrainingControls
        config={trainingState.config}
        onConfigChange={handleConfigChange}
        memoryThreshold={0.9}
        performanceTarget={200} // 200ms target per batch
      />

      {/* Training progress display */}
      <TrainingProgress
        modelId={modelId}
        onMemoryPressure={(usage) => {
          setIsMemoryWarning(true);
          onMemoryPressure?.();
        }}
        onWebGLContextLoss={() => {
          handleContextLoss();
          onContextLoss?.();
        }}
      />
    </TrainingContainer>
  );
});

// Display name for debugging
Training.displayName = 'Training';

export default Training;