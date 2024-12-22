import React, { useCallback, useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import { TextField } from '@mui/material';
import { COMPONENT_SIZES } from '../../constants/ui';
import { validateModelConfig } from '../../lib/utils/validation';

/**
 * Props interface for the enhanced Input component with ML-specific validation
 */
interface InputProps {
  name: string;
  value: string | number;
  onChange: (value: string | number) => void;
  label: string;
  error?: string;
  type?: 'text' | 'number' | 'password';
  validationType?: 'modelConfig' | 'general';
  ariaLabel?: string;
  disabled?: boolean;
  required?: boolean;
  placeholder?: string;
}

/**
 * Styled TextField component with ML-specific visual enhancements
 * @version 1.0.0
 */
const StyledInput = styled(TextField)(({ theme }) => ({
  height: COMPONENT_SIZES.INPUT_HEIGHT,
  width: '100%',
  marginBottom: theme.spacing(2),

  // ML-specific input styling
  '& .MuiInputBase-root': {
    backgroundColor: theme.palette.background.paper,
    transition: theme.transitions.create(['border-color', 'box-shadow']),
    
    '&.Mui-focused': {
      boxShadow: `0 0 0 2px ${theme.palette.primary.main}25`,
    },
    
    '&.Mui-error': {
      boxShadow: `0 0 0 2px ${theme.palette.error.main}25`,
    }
  },

  // Enhanced validation state styling
  '& .MuiFormHelperText-root': {
    marginLeft: 0,
    '&.Mui-error': {
      color: theme.palette.error.main,
      fontWeight: 500
    }
  },

  // Accessibility focus indicators
  '& .MuiOutlinedInput-root': {
    '&:focus-within': {
      outline: `2px solid ${theme.palette.primary.main}`,
      outlineOffset: 2
    }
  },

  // Disabled state with ML context
  '&.Mui-disabled': {
    opacity: 0.7,
    cursor: 'not-allowed',
    backgroundColor: theme.palette.action.disabledBackground
  }
}));

/**
 * Enhanced Input component with ML-specific validation and accessibility features
 * @param props - Input component props
 * @returns JSX.Element
 */
const Input: React.FC<InputProps> = ({
  name,
  value,
  onChange,
  label,
  error,
  type = 'text',
  validationType = 'general',
  ariaLabel,
  disabled = false,
  required = false,
  placeholder
}) => {
  // Local state for validation
  const [localError, setLocalError] = useState<string | undefined>(error);
  const [isValidating, setIsValidating] = useState(false);

  /**
   * Debounced validation handler for ML-specific inputs
   */
  const validateInput = useCallback(async (value: string | number) => {
    if (validationType === 'modelConfig') {
      setIsValidating(true);
      try {
        const result = await validateModelConfig({ [name]: value });
        if (!result.isValid) {
          setLocalError(result.errors[0]);
        } else {
          setLocalError(undefined);
        }
      } catch (err) {
        setLocalError('Validation error occurred');
      }
      setIsValidating(false);
    }
  }, [name, validationType]);

  /**
   * Enhanced change handler with type conversion and validation
   */
  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    let processedValue: string | number = newValue;

    // Handle numeric conversion for model parameters
    if (type === 'number') {
      processedValue = newValue === '' ? '' : Number(newValue);
      if (typeof processedValue === 'number' && isNaN(processedValue)) {
        setLocalError('Please enter a valid number');
        return;
      }
    }

    // Clear previous validation errors
    setLocalError(undefined);
    
    // Trigger validation for ML-specific inputs
    if (validationType === 'modelConfig') {
      validateInput(processedValue);
    }

    onChange(processedValue);
  }, [onChange, type, validationType, validateInput]);

  // Update local error state when prop changes
  useEffect(() => {
    setLocalError(error);
  }, [error]);

  return (
    <StyledInput
      name={name}
      value={value}
      onChange={handleChange}
      label={label}
      error={!!localError}
      helperText={localError}
      type={type}
      disabled={disabled || isValidating}
      required={required}
      placeholder={placeholder}
      variant="outlined"
      fullWidth
      // Enhanced accessibility attributes
      aria-label={ariaLabel || label}
      aria-invalid={!!localError}
      aria-required={required}
      aria-busy={isValidating}
      // ML-specific data attributes
      data-validation-type={validationType}
      data-processing={isValidating}
      InputLabelProps={{
        shrink: true,
        required
      }}
      // Additional props for ML context
      InputProps={{
        'aria-describedby': localError ? `${name}-error` : undefined,
        endAdornment: isValidating ? (
          <div className="validation-indicator" aria-label="Validating input">
            {/* Add loading indicator if needed */}
          </div>
        ) : null
      }}
    />
  );
};

export default Input;