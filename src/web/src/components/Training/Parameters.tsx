import React, { useState, useEffect, useCallback } from 'react';
import { styled } from '@mui/material/styles';
import { Tooltip, Alert } from '@mui/material';
import Input from '../Common/Input';
import Select from '../Common/Select';
import { TrainingConfig, getTrainingConfig, TRAINING_CONSTRAINTS } from '../../config/training';
import { useTraining } from '../../hooks/useTraining';

// Props interface for the TrainingParameters component
interface TrainingParametersProps {
  onConfigChange: (config: TrainingConfig) => void;
  disabled: boolean;
}

// Interface for parameter validation state
interface ParameterValidationState {
  isValid: boolean;
  errors: Record<string, string>;
}

// Styled container for parameters form
const StyledContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: '16px',
  padding: '16px',
  position: 'relative',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
  '& .MuiAlert-root': {
    marginBottom: theme.spacing(2)
  }
}));

/**
 * Enhanced training parameters component with real-time validation and memory monitoring
 */
const TrainingParameters: React.FC<TrainingParametersProps> = ({
  onConfigChange,
  disabled
}) => {
  // Training hook for state management and monitoring
  const { trainingState, memoryStats, browserSupport } = useTraining();

  // Local state for parameter values and validation
  const [config, setConfig] = useState<TrainingConfig>(() => {
    return getTrainingConfig({
      modelConfig: {
        memoryConstraints: {
          maxRAMUsage: 4 * 1024 * 1024 * 1024, // 4GB
          maxGPUMemory: 2 * 1024 * 1024 * 1024, // 2GB
          tensorBufferSize: 256 * 1024 * 1024,   // 256MB
          enableMemoryTracking: true
        }
      }
    }, {
      totalGPUMemory: 4 * 1024 * 1024 * 1024,
      availableGPUMemory: 3 * 1024 * 1024 * 1024,
      totalRAM: 4 * 1024 * 1024 * 1024,
      browserMemoryLimit: 4 * 1024 * 1024 * 1024
    });
  });

  // Validation state
  const [validation, setValidation] = useState<ParameterValidationState>({
    isValid: true,
    errors: {}
  });

  // Memory warning state
  const [memoryWarning, setMemoryWarning] = useState<string | null>(null);

  /**
   * Validates parameter values against constraints
   */
  const validateParameters = useCallback((newConfig: TrainingConfig): ParameterValidationState => {
    const errors: Record<string, string> = {};

    // Batch size validation
    if (newConfig.batchSize < TRAINING_CONSTRAINTS.MIN_BATCH_SIZE || 
        newConfig.batchSize > TRAINING_CONSTRAINTS.MAX_BATCH_SIZE) {
      errors.batchSize = `Batch size must be between ${TRAINING_CONSTRAINTS.MIN_BATCH_SIZE} and ${TRAINING_CONSTRAINTS.MAX_BATCH_SIZE}`;
    }

    // Learning rate validation
    if (newConfig.learningRate < TRAINING_CONSTRAINTS.MIN_LEARNING_RATE || 
        newConfig.learningRate > TRAINING_CONSTRAINTS.MAX_LEARNING_RATE) {
      errors.learningRate = `Learning rate must be between ${TRAINING_CONSTRAINTS.MIN_LEARNING_RATE} and ${TRAINING_CONSTRAINTS.MAX_LEARNING_RATE}`;
    }

    // Epochs validation
    if (newConfig.epochs <= 0 || newConfig.epochs > TRAINING_CONSTRAINTS.MAX_EPOCHS) {
      errors.epochs = `Epochs must be between 1 and ${TRAINING_CONSTRAINTS.MAX_EPOCHS}`;
    }

    return {
      isValid: Object.keys(errors).length === 0,
      errors
    };
  }, []);

  /**
   * Checks memory impact of parameter changes
   */
  const checkMemoryImpact = useCallback((newConfig: TrainingConfig) => {
    const estimatedMemory = newConfig.batchSize * TRAINING_CONSTRAINTS.MIN_MEMORY_PER_BATCH;
    const warning = estimatedMemory > memoryStats.heapUsed * 0.8 
      ? 'Warning: Selected parameters may cause high memory usage'
      : null;
    setMemoryWarning(warning);
  }, [memoryStats.heapUsed]);

  /**
   * Handles parameter changes with validation and memory monitoring
   */
  const handleParameterChange = useCallback((paramName: string, value: string | number) => {
    const newConfig = {
      ...config,
      [paramName]: typeof value === 'string' ? parseFloat(value) : value
    };

    // Validate new configuration
    const validationResult = validateParameters(newConfig);
    setValidation(validationResult);

    // Check memory impact
    checkMemoryImpact(newConfig);

    // Update configuration if valid
    if (validationResult.isValid) {
      setConfig(newConfig);
      onConfigChange(newConfig);
    }
  }, [config, onConfigChange, validateParameters, checkMemoryImpact]);

  // Monitor browser compatibility
  useEffect(() => {
    if (!browserSupport.isCompatible) {
      setValidation(prev => ({
        ...prev,
        errors: {
          ...prev.errors,
          browser: `Browser compatibility issue: ${browserSupport.warnings.join(', ')}`
        }
      }));
    }
  }, [browserSupport]);

  return (
    <StyledContainer>
      {/* Browser compatibility warning */}
      {!browserSupport.isCompatible && (
        <Alert severity="warning">
          Browser compatibility issues detected. Some features may not work optimally.
        </Alert>
      )}

      {/* Memory usage warning */}
      {memoryWarning && (
        <Alert severity="warning">
          {memoryWarning}
        </Alert>
      )}

      {/* Batch size input */}
      <Tooltip title={`Range: ${TRAINING_CONSTRAINTS.MIN_BATCH_SIZE}-${TRAINING_CONSTRAINTS.MAX_BATCH_SIZE}`}>
        <div>
          <Input
            name="batchSize"
            value={config.batchSize}
            onChange={(value) => handleParameterChange('batchSize', value)}
            label="Batch Size"
            type="number"
            error={validation.errors.batchSize}
            disabled={disabled}
            required
          />
        </div>
      </Tooltip>

      {/* Learning rate input */}
      <Tooltip title={`Range: ${TRAINING_CONSTRAINTS.MIN_LEARNING_RATE}-${TRAINING_CONSTRAINTS.MAX_LEARNING_RATE}`}>
        <div>
          <Input
            name="learningRate"
            value={config.learningRate}
            onChange={(value) => handleParameterChange('learningRate', value)}
            label="Learning Rate"
            type="number"
            error={validation.errors.learningRate}
            disabled={disabled}
            required
          />
        </div>
      </Tooltip>

      {/* Epochs input */}
      <Tooltip title={`Maximum: ${TRAINING_CONSTRAINTS.MAX_EPOCHS}`}>
        <div>
          <Input
            name="epochs"
            value={config.epochs}
            onChange={(value) => handleParameterChange('epochs', value)}
            label="Epochs"
            type="number"
            error={validation.errors.epochs}
            disabled={disabled}
            required
          />
        </div>
      </Tooltip>

      {/* Optimizer selection */}
      <Select
        options={['adam', 'sgd', 'rmsprop']}
        value={config.optimizerType}
        onChange={(value) => handleParameterChange('optimizerType', value)}
        label="Optimizer"
        disabled={disabled}
      />

      {/* Memory usage display */}
      <Alert severity="info">
        Memory Usage: {Math.round(memoryStats.heapUsed / (1024 * 1024))}MB / {Math.round(memoryStats.heapTotal / (1024 * 1024))}MB
        {memoryStats.gpuMemory && ` (GPU: ${Math.round(memoryStats.gpuMemory / (1024 * 1024))}MB)`}
      </Alert>
    </StyledContainer>
  );
};

export default TrainingParameters;