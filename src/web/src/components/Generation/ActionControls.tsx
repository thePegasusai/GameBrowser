import React, { useState, useCallback, useEffect } from 'react';
import styled from '@emotion/styled'; // v11.x
import { debounce } from 'lodash'; // v4.x
import { Slider, SliderProps } from '../Common/Slider';
import { useGeneration } from '../../hooks/useGeneration';
import { validateMemoryConstraints } from '../../lib/utils/validation';
import { PERFORMANCE_THRESHOLDS } from '../../constants/model';

/**
 * Props interface for ActionControls component with memory management
 */
interface ActionControlsProps {
  onActionChange: (actionType: string, value: number) => void;
  disabled: boolean;
  currentActions: Record<string, number>;
  memoryConfig: {
    maxMemoryUsage: number;
    warningThreshold: number;
  };
}

/**
 * State interface for action values with tensor tracking
 */
interface ActionState {
  movement: number;
  rotation: number;
  jump: boolean;
  attack: boolean;
  activeTensors: Set<string>;
}

/**
 * Styled components with performance monitoring
 */
const ControlsContainer = styled.div<{ disabled?: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background-color: var(--background-secondary);
  border-radius: 8px;
  position: relative;
  opacity: ${props => props.disabled ? 0.5 : 1};
  transition: opacity 0.2s ease;
`;

const ActionGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  position: relative;
`;

const ActionButton = styled.button<{ active: boolean }>`
  padding: 8px 16px;
  border-radius: 4px;
  border: none;
  background-color: ${props => props.active ? 'var(--primary)' : 'var(--background-tertiary)'};
  color: var(--text-primary);
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  transition: background-color 0.2s ease;

  &:hover:not(:disabled) {
    background-color: var(--primary-hover);
  }
`;

const MemoryIndicator = styled.div<{ warning?: boolean }>`
  position: absolute;
  top: 4px;
  right: 4px;
  font-size: 12px;
  color: ${props => props.warning ? 'var(--error)' : 'var(--text-secondary)'};
`;

/**
 * ActionControls component for managing game actions during video generation
 */
export const ActionControls: React.FC<ActionControlsProps> = ({
  onActionChange,
  disabled,
  currentActions,
  memoryConfig
}) => {
  // State management with memory tracking
  const [state, setState] = useState<ActionState>({
    movement: 0,
    rotation: 0,
    jump: false,
    attack: false,
    activeTensors: new Set()
  });

  const [memoryWarning, setMemoryWarning] = useState(false);
  const { generateFrame, isGenerating } = useGeneration();

  // Debounced memory validation
  const checkMemoryUsage = useCallback(
    debounce(async () => {
      const result = await validateMemoryConstraints(memoryConfig.maxMemoryUsage);
      setMemoryWarning(!result.canAllocate);
    }, 1000),
    [memoryConfig.maxMemoryUsage]
  );

  /**
   * Handles movement control changes with memory optimization
   */
  const handleMovementChange = useCallback(async (value: number) => {
    if (disabled) return;

    const startTime = performance.now();
    setState(prev => ({ ...prev, movement: value }));
    onActionChange('movement', value);

    // Track performance
    const processingTime = performance.now() - startTime;
    if (processingTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
      console.warn(`Movement processing exceeded time threshold: ${processingTime}ms`);
    }

    await checkMemoryUsage();
  }, [disabled, onActionChange, checkMemoryUsage]);

  /**
   * Handles rotation control changes with memory management
   */
  const handleRotationChange = useCallback(async (value: number) => {
    if (disabled) return;

    const startTime = performance.now();
    setState(prev => ({ ...prev, rotation: value }));
    onActionChange('rotation', value);

    // Track performance
    const processingTime = performance.now() - startTime;
    if (processingTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
      console.warn(`Rotation processing exceeded time threshold: ${processingTime}ms`);
    }

    await checkMemoryUsage();
  }, [disabled, onActionChange, checkMemoryUsage]);

  /**
   * Handles discrete action toggles with memory tracking
   */
  const handleActionToggle = useCallback(async (actionType: 'jump' | 'attack') => {
    if (disabled) return;

    setState(prev => ({
      ...prev,
      [actionType]: !prev[actionType]
    }));
    onActionChange(actionType, state[actionType] ? 0 : 1);
    await checkMemoryUsage();
  }, [disabled, state, onActionChange, checkMemoryUsage]);

  // Monitor memory usage
  useEffect(() => {
    const interval = setInterval(checkMemoryUsage, 5000);
    return () => clearInterval(interval);
  }, [checkMemoryUsage]);

  return (
    <ControlsContainer disabled={disabled}>
      <MemoryIndicator warning={memoryWarning}>
        {memoryWarning ? 'Memory Warning' : 'Memory OK'}
      </MemoryIndicator>

      <ActionGroup>
        <Slider
          label="Movement"
          min={-1}
          max={1}
          step={0.1}
          value={state.movement}
          onChange={handleMovementChange}
          disabled={disabled}
          validateMemory={true}
        />
      </ActionGroup>

      <ActionGroup>
        <Slider
          label="Camera Rotation"
          min={-180}
          max={180}
          step={5}
          value={state.rotation}
          onChange={handleRotationChange}
          disabled={disabled}
          validateMemory={true}
          unit="°"
        />
      </ActionGroup>

      <ActionGroup>
        <ActionButton
          active={state.jump}
          onClick={() => handleActionToggle('jump')}
          disabled={disabled}
        >
          Jump
        </ActionButton>
        <ActionButton
          active={state.attack}
          onClick={() => handleActionToggle('attack')}
          disabled={disabled}
        >
          Attack
        </ActionButton>
      </ActionGroup>
    </ControlsContainer>
  );
};

export default ActionControls;