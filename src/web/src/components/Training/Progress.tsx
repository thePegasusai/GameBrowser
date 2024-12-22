import React, { useEffect, useMemo } from 'react';
import { styled } from '@mui/material/styles';
import { Box, Typography, Alert } from '@mui/material';
import Progress from '../Common/Progress';
import { useTraining } from '../../hooks/useTraining';
import { THEME_COLORS } from '../../constants/ui';

/**
 * Props interface for the TrainingProgress component
 */
interface TrainingProgressProps {
  modelId: string;
  className?: string;
  onMemoryPressure?: (usage: number) => void;
  onWebGLContextLoss?: () => void;
}

/**
 * Styled container for the progress component with enhanced visual feedback
 */
const ProgressContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: '16px',
  padding: '16px',
  borderRadius: '8px',
  backgroundColor: 'rgba(0, 0, 0, 0.05)',
  position: 'relative',
  overflow: 'hidden'
}));

/**
 * Styled container for training metrics display
 */
const MetricsContainer = styled(Box)(({ theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
  gap: '16px',
  alignItems: 'center'
}));

/**
 * Determines progress status based on training state and system metrics
 */
const getProgressStatus = (
  trainingState: any,
  memoryUsage: any,
  webglContext: any
): 'default' | 'error' | 'warning' | 'success' => {
  // Check for critical errors
  if (!webglContext.isValid) {
    return 'error';
  }

  // Check memory pressure
  if (memoryUsage.utilizationPercentage > 0.9) {
    return 'warning';
  }

  // Check training state
  if (trainingState.status === 'error') {
    return 'error';
  }

  if (trainingState.status === 'completed') {
    return 'success';
  }

  return 'default';
};

/**
 * Enhanced training progress component with memory and WebGL monitoring
 */
export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  modelId,
  className,
  onMemoryPressure,
  onWebGLContextLoss
}) => {
  // Get training state and monitoring data
  const { trainingState, memoryStats, browserSupport } = useTraining({ modelId });

  // Monitor memory pressure
  useEffect(() => {
    if (memoryStats.utilizationPercentage > 0.9 && onMemoryPressure) {
      onMemoryPressure(memoryStats.utilizationPercentage);
    }
  }, [memoryStats.utilizationPercentage, onMemoryPressure]);

  // Monitor WebGL context
  useEffect(() => {
    if (!browserSupport.webglSupported && onWebGLContextLoss) {
      onWebGLContextLoss();
    }
  }, [browserSupport.webglSupported, onWebGLContextLoss]);

  // Calculate progress status
  const progressStatus = useMemo(() => 
    getProgressStatus(trainingState, memoryStats, browserSupport),
    [trainingState, memoryStats, browserSupport]
  );

  // Format memory usage for display
  const formattedMemory = useMemo(() => ({
    used: `${(memoryStats.heapUsed / (1024 * 1024)).toFixed(2)} MB`,
    total: `${(memoryStats.heapTotal / (1024 * 1024)).toFixed(2)} MB`,
    gpu: memoryStats.gpuMemory ? 
      `${(memoryStats.gpuMemory / (1024 * 1024)).toFixed(2)} MB` : 
      'N/A'
  }), [memoryStats]);

  return (
    <ProgressContainer className={className}>
      {/* Status alerts */}
      {progressStatus === 'error' && (
        <Alert severity="error">
          Training error occurred. Check console for details.
        </Alert>
      )}
      {progressStatus === 'warning' && (
        <Alert severity="warning">
          High memory usage detected. Performance may be affected.
        </Alert>
      )}

      {/* Progress bar */}
      <Progress
        value={trainingState.progress}
        variant={trainingState.isTraining ? 'determinate' : 'indeterminate'}
        status={progressStatus}
        aria-label="Training progress"
      />

      {/* Training metrics */}
      <MetricsContainer>
        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            Epoch
          </Typography>
          <Typography variant="body1">
            {trainingState.currentEpoch} / {trainingState.totalEpochs}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            Batch
          </Typography>
          <Typography variant="body1">
            {trainingState.currentBatch}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            Loss
          </Typography>
          <Typography variant="body1">
            {trainingState.loss.toFixed(4)}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            Step Time
          </Typography>
          <Typography variant="body1">
            {trainingState.stepTime.toFixed(2)}ms
          </Typography>
        </Box>
      </MetricsContainer>

      {/* Memory usage */}
      <MetricsContainer>
        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            RAM Usage
          </Typography>
          <Typography variant="body1">
            {formattedMemory.used} / {formattedMemory.total}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            GPU Memory
          </Typography>
          <Typography variant="body1">
            {formattedMemory.gpu}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="textSecondary">
            WebGL Status
          </Typography>
          <Typography 
            variant="body1"
            color={browserSupport.webglSupported ? 'success.main' : 'error.main'}
          >
            {browserSupport.webglSupported ? 'Active' : 'Lost'}
          </Typography>
        </Box>
      </MetricsContainer>
    </ProgressContainer>
  );
};

export default TrainingProgress;