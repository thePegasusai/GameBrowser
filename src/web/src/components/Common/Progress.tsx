import React, { useMemo } from 'react';
import { styled } from '@mui/material/styles';
import LinearProgress from '@mui/material/LinearProgress';
import { THEME_COLORS, COMPONENT_SIZES } from '../../constants/ui';

/**
 * Props interface for the Progress component
 * Provides type safety and documentation for component properties
 */
interface ProgressProps {
  /** Current progress value between 0-100 */
  value?: number;
  /** Progress bar variant - determinate shows specific progress, indeterminate shows loading */
  variant?: 'determinate' | 'indeterminate';
  /** Status affecting color and accessibility label */
  status?: 'default' | 'error' | 'success';
  /** Optional custom height in pixels */
  height?: number;
  /** Accessibility label for screen readers */
  ariaLabel?: string;
}

/**
 * Memoized function to determine progress bar color based on status
 * @param status - Current progress status
 * @returns Theme color code based on status
 */
const getProgressColor = (status?: 'default' | 'error' | 'success'): string => {
  switch (status) {
    case 'error':
      return THEME_COLORS.ERROR;
    case 'success':
      return THEME_COLORS.SUCCESS;
    default:
      return THEME_COLORS.PRIMARY;
  }
};

/**
 * Styled LinearProgress component with enhanced visual features
 * Supports custom height and dynamic color based on status
 */
const StyledProgress = styled(LinearProgress, {
  shouldForwardProp: (prop) => prop !== 'status' && prop !== 'height',
})<{ status?: string; height?: number }>(({ status, height }) => ({
  height: `${height || COMPONENT_SIZES.PROGRESS_HEIGHT}px`,
  borderRadius: `${height || COMPONENT_SIZES.PROGRESS_HEIGHT}px`,
  backgroundColor: 'rgba(0, 0, 0, 0.1)',
  transition: 'all 0.3s ease-in-out',
  '& .MuiLinearProgress-bar': {
    backgroundColor: getProgressColor(status as 'default' | 'error' | 'success'),
    transition: 'background-color 0.3s ease-in-out',
  },
}));

/**
 * Progress component for displaying training, generation, and processing progress
 * Supports both determinate and indeterminate states with customizable appearance
 * 
 * @example
 * // Determinate progress with success status
 * <Progress value={75} status="success" ariaLabel="Training progress" />
 * 
 * @example
 * // Indeterminate progress for loading states
 * <Progress variant="indeterminate" ariaLabel="Loading..." />
 */
export const Progress: React.FC<ProgressProps> = ({
  value = 0,
  variant = 'determinate',
  status = 'default',
  height,
  ariaLabel,
}) => {
  // Memoize the accessibility label based on status and progress
  const accessibilityLabel = useMemo(() => {
    if (ariaLabel) return ariaLabel;
    
    const baseLabel = variant === 'indeterminate' ? 'Loading' : `Progress: ${value}%`;
    switch (status) {
      case 'error':
        return `Error - ${baseLabel}`;
      case 'success':
        return `Complete - ${baseLabel}`;
      default:
        return baseLabel;
    }
  }, [ariaLabel, status, value, variant]);

  return (
    <StyledProgress
      variant={variant}
      value={value}
      status={status}
      height={height}
      aria-label={accessibilityLabel}
      role="progressbar"
      aria-valuenow={variant === 'determinate' ? value : undefined}
      aria-valuemin={0}
      aria-valuemax={100}
    />
  );
};

// Default export for convenient importing
export default Progress;