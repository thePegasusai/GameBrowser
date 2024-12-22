/**
 * Training controls component with memory monitoring and performance optimization
 * Provides real-time feedback and control over model training process
 * @version 1.0.0
 */

import React, { useCallback, useEffect, useState } from 'react'; // react@18.x
import { Box, Stack, Alert, CircularProgress } from '@mui/material'; // @mui/material@5.x
import CustomButton from '../Common/Button';
import { useTraining } from '../../hooks/useTraining';
import { ModelState } from '../../types/model';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../constants/model';

interface TrainingControlsProps {
  config: TrainingConfig;
  onConfigChange: (config: TrainingConfig) => void;
  memoryThreshold: number;
  performanceTarget: number;
}

/**
 * Training controls component with memory-aware state management
 */
const TrainingControls: React.FC<TrainingControlsProps> = ({
  config,
  onConfigChange,
  memoryThreshold = MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD,
  performanceTarget = PERFORMANCE_THRESHOLDS.MAX_TRAINING_STEP_TIME
}) => {
  // Training state management
  const {
    trainingState,
    memoryStats,
    browserSupport,
    startTraining,
    pauseTraining,
    resumeTraining
  } = useTraining();

  // Local state for UI feedback
  const [error, setError] = useState<string | null>(null);
  const [lastPerformanceCheck, setLastPerformanceCheck] = useState(0);

  /**
   * Handles start training button click with memory validation
   */
  const handleStartClick = useCallback(async (event: React.MouseEvent) => {
    try {
      // Validate browser compatibility
      if (!browserSupport.isCompatible) {
        throw new Error(`Browser not compatible: ${browserSupport.warnings.join(', ')}`);
      }

      // Check memory availability
      if (memoryStats.utilizationPercentage > memoryThreshold) {
        throw new Error('Insufficient memory available for training');
      }

      await startTraining();
    } catch (error) {
      setError(error.message);
    }
  }, [browserSupport, memoryStats, memoryThreshold, startTraining]);

  /**
   * Monitors training performance and memory usage
   */
  useEffect(() => {
    const checkPerformance = () => {
      const now = Date.now();
      if (now - lastPerformanceCheck > PERFORMANCE_THRESHOLDS.PERFORMANCE_CHECK_INTERVAL) {
        // Check step time
        if (trainingState.stepTime > performanceTarget) {
          setError(`Training step time ${trainingState.stepTime}ms exceeds target ${performanceTarget}ms`);
        }
        
        // Check memory usage
        if (memoryStats.utilizationPercentage > memoryThreshold) {
          setError('Memory usage exceeds threshold, consider reducing batch size');
        }

        setLastPerformanceCheck(now);
      }
    };

    const intervalId = setInterval(checkPerformance, PERFORMANCE_THRESHOLDS.PERFORMANCE_CHECK_INTERVAL);
    return () => clearInterval(intervalId);
  }, [trainingState, memoryStats, lastPerformanceCheck, performanceTarget, memoryThreshold]);

  /**
   * Renders training progress indicator
   */
  const renderProgress = useCallback(() => {
    if (!trainingState.isTraining) return null;

    return (
      <Box display="flex" alignItems="center" gap={2}>
        <CircularProgress
          variant="determinate"
          value={trainingState.progress * 100}
          size={24}
        />
        <Box>
          {`Epoch ${trainingState.currentEpoch + 1}, Batch ${trainingState.currentBatch + 1}`}
        </Box>
      </Box>
    );
  }, [trainingState]);

  /**
   * Renders memory usage indicator
   */
  const renderMemoryStatus = useCallback(() => {
    const usagePercentage = memoryStats.utilizationPercentage * 100;
    const color = usagePercentage > 90 ? 'error' : usagePercentage > 70 ? 'warning' : 'success';

    return (
      <Box>
        <CircularProgress
          variant="determinate"
          value={usagePercentage}
          color={color}
          size={24}
        />
        <Box>{`Memory: ${Math.round(usagePercentage)}%`}</Box>
      </Box>
    );
  }, [memoryStats]);

  return (
    <Box>
      <Stack spacing={2}>
        {/* Error display */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Training controls */}
        <Stack direction="row" spacing={2}>
          <CustomButton
            variant="contained"
            color="ml-primary"
            onClick={handleStartClick}
            disabled={trainingState.isTraining || !browserSupport.isCompatible}
            loading={trainingState.status === ModelState.LOADING}
            ariaLabel="Start training"
          >
            Start Training
          </CustomButton>

          <CustomButton
            variant="outlined"
            color="ml-secondary"
            onClick={trainingState.status === ModelState.TRAINING ? pauseTraining : resumeTraining}
            disabled={!trainingState.isTraining}
            ariaLabel={trainingState.status === ModelState.TRAINING ? "Pause training" : "Resume training"}
          >
            {trainingState.status === ModelState.TRAINING ? 'Pause' : 'Resume'}
          </CustomButton>
        </Stack>

        {/* Progress and metrics */}
        <Stack direction="row" spacing={4} alignItems="center">
          {renderProgress()}
          {renderMemoryStatus()}
          
          {/* Performance metrics */}
          <Box>
            <Box>{`Step Time: ${trainingState.stepTime.toFixed(1)}ms`}</Box>
            <Box>{`Loss: ${trainingState.loss.toFixed(4)}`}</Box>
          </Box>
        </Stack>

        {/* Browser compatibility warning */}
        {!browserSupport.isCompatible && (
          <Alert severity="warning">
            {`Browser compatibility issues: ${browserSupport.warnings.join(', ')}`}
          </Alert>
        )}
      </Stack>
    </Box>
  );
};

export default TrainingControls;