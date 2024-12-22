import React, { useCallback } from 'react';
import { 
  Select as MuiSelect, 
  MenuItem, 
  FormControl,
  FormHelperText,
  SelectChangeEvent,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { THEME_COLORS, COMPONENT_SIZES } from '../../constants/ui';

// Interface for component props with strict typing
interface SelectProps {
  /** Array of options to display in the select menu */
  options: string[];
  /** Current selected value(s) */
  value: string | string[];
  /** Callback function when selection changes */
  onChange: (value: string | string[]) => void;
  /** Label text for the select field */
  label?: string;
  /** Enable multiple selection mode */
  multiple?: boolean;
  /** Error state for validation feedback */
  error?: boolean;
  /** Helper text for additional context or error messages */
  helperText?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Custom width in pixels */
  width?: number;
  /** ARIA label for accessibility */
  'aria-label'?: string;
  /** Unique identifier */
  id?: string;
}

// Styled wrapper for MUI Select with enhanced customization
const StyledSelect = styled(MuiSelect)(({ theme, error, width }) => ({
  height: COMPONENT_SIZES.INPUT_HEIGHT,
  width: width || '100%',
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: error ? THEME_COLORS.ERROR : 'rgba(0, 0, 0, 0.23)',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: error ? THEME_COLORS.ERROR : THEME_COLORS.PRIMARY,
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: error ? THEME_COLORS.ERROR : THEME_COLORS.PRIMARY,
    borderWidth: 2,
  },
  '& .MuiSelect-select': {
    height: COMPONENT_SIZES.INPUT_HEIGHT - 2,
    padding: '0 14px',
    display: 'flex',
    alignItems: 'center',
  },
  '&.Mui-disabled': {
    backgroundColor: theme.palette.action.disabledBackground,
    cursor: 'not-allowed',
  },
}));

// Styled MenuItem for consistent dropdown styling
const StyledMenuItem = styled(MenuItem)(({ theme }) => ({
  height: COMPONENT_SIZES.INPUT_HEIGHT,
  '&.Mui-selected': {
    backgroundColor: `${THEME_COLORS.PRIMARY}14`,
  },
  '&.Mui-selected:hover': {
    backgroundColor: `${THEME_COLORS.PRIMARY}20`,
  },
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

/**
 * Enhanced Select component with Material Design styling and accessibility features
 * @param props - SelectProps interface properties
 * @returns JSX.Element - Rendered select component
 */
const Select: React.FC<SelectProps> = ({
  options,
  value,
  onChange,
  label,
  multiple = false,
  error = false,
  helperText,
  disabled = false,
  width,
  'aria-label': ariaLabel,
  id,
}) => {
  // Enhanced change handler with type safety
  const handleChange = useCallback((event: SelectChangeEvent<unknown>) => {
    const newValue = event.target.value;
    onChange(multiple ? (newValue as string[]) : (newValue as string));
  }, [multiple, onChange]);

  // Render helper text if provided
  const renderHelperText = () => {
    if (!helperText) return null;
    return (
      <FormHelperText error={error}>
        {helperText}
      </FormHelperText>
    );
  };

  return (
    <FormControl 
      error={error}
      disabled={disabled}
      sx={{ width: width || '100%' }}
    >
      <StyledSelect
        id={id}
        value={value}
        onChange={handleChange}
        multiple={multiple}
        error={error}
        width={width}
        aria-label={ariaLabel || label}
        // Enhanced keyboard navigation support
        MenuProps={{
          PaperProps: {
            style: {
              maxHeight: COMPONENT_SIZES.INPUT_HEIGHT * 8,
            },
          },
          anchorOrigin: {
            vertical: 'bottom',
            horizontal: 'left',
          },
          transformOrigin: {
            vertical: 'top',
            horizontal: 'left',
          },
          // Improved screen reader support
          getContentAnchorEl: null,
        }}
        // ARIA attributes for accessibility
        aria-describedby={helperText ? `${id}-helper-text` : undefined}
      >
        {options.map((option) => (
          <StyledMenuItem
            key={option}
            value={option}
            // Enhanced ARIA support for options
            role="option"
            aria-selected={multiple ? 
              (value as string[])?.includes(option) : 
              value === option
            }
          >
            {option}
          </StyledMenuItem>
        ))}
      </StyledSelect>
      {renderHelperText()}
    </FormControl>
  );
};

export default Select;