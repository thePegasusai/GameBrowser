/**
 * Global theme configuration for browser-based video game diffusion model interface
 * Implements Material Design principles with ML-specific component support
 * @version 1.0.0
 */

import { createTheme, ThemeOptions, Theme } from '@mui/material'; // @mui/material@5.x
import { THEME_COLORS } from '../constants/ui';

/**
 * Extended theme options interface for ML-specific styling
 */
interface CustomThemeOptions extends ThemeOptions {
  components?: {
    MLProgressBar?: {
      styleOverrides: {
        root: {
          height: number;
          borderRadius: number;
        };
      };
    };
    MLDataDisplay?: {
      styleOverrides: {
        root: {
          fontFamily: string;
        };
      };
    };
  };
}

/**
 * Typography configuration with technical display variants
 */
const TYPOGRAPHY = {
  fontFamily: "'Roboto Mono', 'Roboto', 'Helvetica', 'Arial', sans-serif",
  h1: {
    fontSize: '2.5rem',
    fontWeight: 500,
    letterSpacing: '-0.01562em',
  },
  h2: {
    fontSize: '2rem',
    fontWeight: 500,
    letterSpacing: '-0.00833em',
  },
  body1: {
    fontSize: '1rem',
    lineHeight: 1.5,
    letterSpacing: '0.00938em',
  },
  technicalDisplay: {
    fontFamily: "'Roboto Mono', monospace",
    fontSize: '0.875rem',
    lineHeight: 1.43,
    letterSpacing: '0.01071em',
  },
};

/**
 * Extended breakpoints supporting 4K resolutions
 */
const BREAKPOINTS = {
  values: {
    xs: 0,
    sm: 600,
    md: 960,
    lg: 1280,
    xl: 1920,
    xxl: 2560,
    uhd: 3840,
  },
};

/**
 * ML-specific component style overrides
 */
const ML_COMPONENT_VARIANTS = {
  MLProgressBar: {
    styleOverrides: {
      root: {
        height: 8,
        borderRadius: 4,
      },
    },
  },
  MLDataDisplay: {
    styleOverrides: {
      root: {
        fontFamily: "'Roboto Mono', monospace",
      },
    },
  },
};

/**
 * Creates light theme configuration with ML-specific components
 */
export const lightTheme: Theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: THEME_COLORS.PRIMARY,
      contrastText: '#ffffff',
    },
    secondary: {
      main: THEME_COLORS.SECONDARY,
      contrastText: '#ffffff',
    },
    error: {
      main: THEME_COLORS.ERROR,
    },
    success: {
      main: THEME_COLORS.SUCCESS,
    },
    warning: {
      main: THEME_COLORS.WARNING,
    },
    background: {
      default: '#ffffff',
      paper: '#f5f5f5',
    },
    ml: {
      primary: THEME_COLORS.ML_PRIMARY,
      secondary: THEME_COLORS.ML_SECONDARY,
      accent: THEME_COLORS.ML_ACCENT,
    },
  },
  typography: TYPOGRAPHY,
  breakpoints: BREAKPOINTS,
  spacing: (factor: number) => `${8 * factor}px`,
  components: {
    ...ML_COMPONENT_VARIANTS,
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
} as CustomThemeOptions);

/**
 * Creates dark theme configuration with ML-specific components
 */
export const darkTheme: Theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: THEME_COLORS.PRIMARY,
      contrastText: '#ffffff',
    },
    secondary: {
      main: THEME_COLORS.SECONDARY,
      contrastText: '#ffffff',
    },
    error: {
      main: THEME_COLORS.ERROR,
    },
    success: {
      main: THEME_COLORS.SUCCESS,
    },
    warning: {
      main: THEME_COLORS.WARNING,
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    ml: {
      primary: THEME_COLORS.ML_PRIMARY,
      secondary: THEME_COLORS.ML_SECONDARY,
      accent: THEME_COLORS.ML_ACCENT,
    },
  },
  typography: TYPOGRAPHY,
  breakpoints: BREAKPOINTS,
  spacing: (factor: number) => `${8 * factor}px`,
  components: {
    ...ML_COMPONENT_VARIANTS,
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          backgroundColor: '#1e1e1e',
        },
      },
    },
  },
} as CustomThemeOptions);