/**
 * Header component for browser-based video game diffusion model interface
 * Implements Material Design principles with enhanced accessibility and responsive design
 * @version 1.0.0
 */

import React from 'react'; // react@^18.0.0
import { styled } from '@mui/material/styles'; // @mui/material@^5.0.0
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton 
} from '@mui/material'; // @mui/material@^5.0.0
import { 
  Brightness4, 
  Brightness7 
} from '@mui/icons-material'; // @mui/icons-material@^5.0.0

import { CustomButton, ButtonProps } from '../Common/Button';
import { lightTheme } from '../../styles/theme';
import { LAYOUT_SIZES } from '../../constants/ui';

/**
 * Props interface for Header component
 */
export interface HeaderProps {
  /** Application title displayed in header */
  title: string;
  /** Callback function for theme toggle */
  onThemeToggle: (isDark: boolean) => void;
  /** Current theme mode state */
  isDarkMode: boolean;
  /** Accessibility label for theme toggle */
  ariaLabel: string;
}

/**
 * Styled AppBar component with fixed positioning and theme transitions
 */
const StyledAppBar = styled(AppBar)(({ theme }) => ({
  height: LAYOUT_SIZES.HEADER_HEIGHT,
  backgroundColor: theme.palette.primary.main,
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  zIndex: theme.zIndex.appBar,
  transition: theme.transitions.create(['background-color'], {
    duration: 300,
  }),
  boxShadow: theme.shadows[4],
}));

/**
 * Styled Toolbar component with responsive padding and flex layout
 */
const StyledToolbar = styled(Toolbar)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: theme.spacing(0, 3),
  minHeight: LAYOUT_SIZES.HEADER_HEIGHT,
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(0, 1),
  },
}));

/**
 * Header component with theme toggle and responsive design
 */
export const Header: React.FC<HeaderProps> = React.memo(({
  title,
  onThemeToggle,
  isDarkMode,
  ariaLabel,
}) => {
  /**
   * Handles theme toggle with keyboard accessibility
   */
  const handleThemeToggle = React.useCallback(() => {
    onThemeToggle(!isDarkMode);
  }, [isDarkMode, onThemeToggle]);

  return (
    <StyledAppBar elevation={4}>
      <StyledToolbar>
        <Typography
          variant="h6"
          component="h1"
          sx={{
            fontWeight: 500,
            color: theme => theme.palette.primary.contrastText,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            [theme.breakpoints.down('sm')]: {
              fontSize: '1.1rem',
            },
          }}
        >
          {title}
        </Typography>

        <IconButton
          onClick={handleThemeToggle}
          color="inherit"
          aria-label={ariaLabel}
          sx={{
            ml: 1,
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
            },
            transition: theme => 
              theme.transitions.create(['background-color'], {
                duration: 300,
              }),
          }}
        >
          {isDarkMode ? (
            <Brightness7 
              sx={{ 
                transition: theme => 
                  theme.transitions.create(['transform'], {
                    duration: 300,
                  }),
                '&:hover': {
                  transform: 'rotate(180deg)',
                },
              }} 
            />
          ) : (
            <Brightness4 
              sx={{ 
                transition: theme => 
                  theme.transitions.create(['transform'], {
                    duration: 300,
                  }),
                '&:hover': {
                  transform: 'rotate(180deg)',
                },
              }} 
            />
          )}
        </IconButton>
      </StyledToolbar>
    </StyledAppBar>
  );
});

// Display name for debugging
Header.displayName = 'Header';

export default Header;