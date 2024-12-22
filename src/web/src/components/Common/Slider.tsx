import React, { useCallback, useState, useEffect } from 'react';
import styled from '@emotion/styled';
import { debounce } from 'lodash'; // v4.x
import { COMPONENT_SIZES } from '../../constants/ui';
import { validateMemoryConstraints } from '../../lib/utils/validation';

/**
 * Props interface for the Slider component
 */
interface SliderProps {
  min: number;
  max: number;
  value: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  label?: string;
  unit?: string;
  validateMemory?: boolean;
  errorMessage?: string;
  ariaLabel?: string;
}

/**
 * Styled components for the slider with Material Design principles
 */
const SliderContainer = styled.div<{ disabled?: boolean }>`
  display: flex;
  flex-direction: column;
  width: 100%;
  margin-bottom: 16px;
  opacity: ${props => props.disabled ? 0.5 : 1};
  position: relative;
  font-family: 'Roboto', sans-serif;
`;

const SliderLabel = styled.label`
  color: rgba(0, 0, 0, 0.87);
  font-size: 14px;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const SliderValue = styled.span`
  color: rgba(0, 0, 0, 0.6);
  font-size: 12px;
  min-width: 45px;
  text-align: right;
`;

const SliderTrack = styled.div<{ isError?: boolean }>`
  height: ${COMPONENT_SIZES.SLIDER_HEIGHT}px;
  background-color: ${props => props.isError ? '#ffebee' : '#e0e0e0'};
  border-radius: 2px;
  position: relative;
  transition: background-color 0.2s ease;
`;

const SliderProgress = styled.div<{ progress: number; isError?: boolean }>`
  height: 100%;
  background-color: ${props => props.isError ? '#d32f2f' : '#1976d2'};
  border-radius: 2px;
  width: ${props => props.progress}%;
  position: absolute;
  transition: width 0.1s ease, background-color 0.2s ease;
`;

const SliderThumb = styled.div<{ disabled?: boolean; isError?: boolean }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: ${props => props.isError ? '#d32f2f' : '#1976d2'};
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  transition: background-color 0.2s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);

  &:hover {
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
  }

  &:active {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
  }
`;

const ErrorMessage = styled.div`
  color: #d32f2f;
  font-size: 12px;
  margin-top: 4px;
  min-height: 16px;
  transition: opacity 0.2s ease;
`;

/**
 * Formats the slider value with appropriate precision and unit
 */
const formatValue = (value: number, unit?: string): string => {
  const precision = Number.isInteger(value) ? 0 : 2;
  return `${value.toFixed(precision)}${unit ? unit : ''}`;
};

/**
 * Slider component with memory validation support
 */
export const Slider: React.FC<SliderProps> = ({
  min,
  max,
  value,
  step,
  onChange,
  disabled = false,
  label,
  unit,
  validateMemory = false,
  errorMessage,
  ariaLabel
}) => {
  const [isError, setIsError] = useState(false);
  const [localErrorMessage, setLocalErrorMessage] = useState(errorMessage);

  // Calculate progress percentage for the slider
  const progress = ((value - min) / (max - min)) * 100;

  // Debounced memory validation
  const validateValue = useCallback(
    debounce(async (newValue: number) => {
      if (validateMemory) {
        try {
          const result = await validateMemoryConstraints(newValue);
          setIsError(!result.canAllocate);
          setLocalErrorMessage(
            !result.canAllocate
              ? `Exceeds available memory (${result.availableMemory}MB available)`
              : undefined
          );
        } catch (error) {
          setIsError(true);
          setLocalErrorMessage('Memory validation failed');
        }
      }
    }, 300),
    [validateMemory]
  );

  // Handle slider value changes
  const handleSliderChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled) return;

      const newValue = Number(event.target.value);
      const clampedValue = Math.min(Math.max(newValue, min), max);
      const steppedValue = Math.round(clampedValue / step) * step;

      validateValue(steppedValue);
      onChange(steppedValue);
    },
    [disabled, min, max, step, onChange, validateValue]
  );

  // Update error message when prop changes
  useEffect(() => {
    setLocalErrorMessage(errorMessage);
  }, [errorMessage]);

  return (
    <SliderContainer disabled={disabled}>
      {label && (
        <SliderLabel>
          {label}
          <SliderValue>{formatValue(value, unit)}</SliderValue>
        </SliderLabel>
      )}
      <SliderTrack isError={isError}>
        <SliderProgress progress={progress} isError={isError} />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleSliderChange}
          disabled={disabled}
          aria-label={ariaLabel || label}
          style={{
            width: '100%',
            height: '100%',
            opacity: 0,
            cursor: disabled ? 'not-allowed' : 'pointer',
            position: 'absolute',
            top: 0,
            left: 0,
          }}
        />
        <SliderThumb
          style={{ left: `${progress}%` }}
          disabled={disabled}
          isError={isError}
        />
      </SliderTrack>
      <ErrorMessage>
        {localErrorMessage}
      </ErrorMessage>
    </SliderContainer>
  );
};

export default Slider;