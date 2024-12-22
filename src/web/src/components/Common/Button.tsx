/**
 * Enhanced Material-UI Button component with ML-specific optimizations
 * Supports loading states, progress indicators, and ML-specific styling
 * @version 1.0.0
 */

import React from 'react'; // react@18.x
import { Button, CircularProgress } from '@mui/material'; // @mui/material@5.x
import { styled } from '@mui/material/styles'; // @mui/material/styles@5.x
import { lightTheme } from '../../styles/theme';

/**
 * Extended props interface for ML-optimized button component
 */
export interface ButtonProps {
  children: React.ReactNode;
  variant?: 'contained' | 'outlined' | 'text';
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'success' | 'ml-primary' | 'ml-secondary';
  size?: 'small' | 'medium' | 'large';
  fullWidth?: boolean;
  disabled?: boolean;
  loading?: boolean;
  loadingProgress?: number;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  startIcon?: React.ReactNode;
  endIcon?: React.ReactNode;
  ariaLabel?: string;
  testId?: string;
}

/**
 * Enhanced Material-UI Button with ML-specific styling and optimizations
 */
const StyledButton = styled(Button)(({ theme }) => ({
  root: {
    position: 'relative',
    minHeight: 36,
    padding: theme.spacing(1, 2),
    transition: theme.transitions.create(
      ['background-color', 'box-shadow', 'border-color', 'opacity'],
      { duration: 300 }
    ),
  },
  '&.MuiButton-mlPrimary': {
    backgroundColor: lightTheme.palette.ml?.primary,
    color: '#ffffff',
    '&:hover': {
      backgroundColor: lightTheme.palette.ml?.primary,
      opacity: 0.9,
    },
  },
  '&.MuiButton-mlSecondary': {
    backgroundColor: lightTheme.palette.ml?.secondary,
    color: '#ffffff',
    '&:hover': {
      backgroundColor: lightTheme.palette.ml?.secondary,
      opacity: 0.9,
    },
  },
  '&.Mui-disabled': {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  '&.loading': {
    cursor: 'wait',
    '& .MuiButton-startIcon, & .MuiButton-endIcon': {
      opacity: 0,
    },
  },
  '& .loadingIndicator': {
    position: 'absolute',
    left: '50%',
    top: '50%',
    transform: 'translate(-50%, -50%)',
  },
}));

/**
 * Optimized button component with ML-specific states and performance enhancements
 */
const CustomButton = React.memo<ButtonProps>(({
  children,
  variant = 'contained',
  color = 'primary',
  size = 'medium',
  fullWidth = false,
  disabled = false,
  loading = false,
  loadingProgress,
  onClick,
  startIcon,
  endIcon,
  ariaLabel,
  testId,
}) => {
  // Handle ML-specific color variants
  const buttonColor = color.startsWith('ml-') ? undefined : color;
  const mlColor = color.startsWith('ml-') ? color : undefined;

  // Compute effective disabled state
  const isDisabled = disabled || loading;

  return (
    <StyledButton
      variant={variant}
      color={buttonColor}
      size={size}
      fullWidth={fullWidth}
      disabled={isDisabled}
      onClick={onClick}
      className={`${mlColor} ${loading ? 'loading' : ''}`}
      startIcon={!loading && startIcon}
      endIcon={!loading && endIcon}
      aria-label={ariaLabel}
      aria-busy={loading}
      data-testid={testId}
    >
      {children}
      {loading && (
        <div className="loadingIndicator">
          {loadingProgress !== undefined ? (
            <CircularProgress
              size={24}
              value={loadingProgress}
              variant="determinate"
              color="inherit"
            />
          ) : (
            <CircularProgress
              size={24}
              variant="indeterminate"
              color="inherit"
            />
          )}
        </div>
      )}
    </StyledButton>
  );
});

// Display name for debugging
CustomButton.displayName = 'CustomButton';

export default CustomButton;