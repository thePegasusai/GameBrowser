/**
 * Main layout component for browser-based video game diffusion model interface
 * Implements Material Design principles with ML-optimized layout structure
 * @version 1.0.0
 */

import React from 'react'; // ^18.0.0
import { styled, ThemeProvider } from '@mui/material/styles'; // ^5.0.0
import CssBaseline from '@mui/material/CssBaseline'; // ^5.0.0
import useMediaQuery from '@mui/material/useMediaQuery'; // ^5.0.0

// Internal imports
import Header from './Header';
import Footer from './Footer';
import { lightTheme, darkTheme } from '../../styles/theme';

/**
 * Props interface for Layout component
 */
interface LayoutProps {
  children: React.ReactNode;
  version: string;
}

/**
 * Styled container component with responsive layout and theme transitions
 */
const LayoutContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  minHeight: '100vh',
  width: '100%',
  overflowX: 'hidden',
  position: 'relative',
  transition: theme.transitions.create(['background-color'], {
    duration: 300,
  }),
  backgroundColor: theme.palette.background.default,
  [theme.breakpoints.up('uhd')]: {
    maxWidth: '3840px',
    margin: '0 auto',
  },
}));

/**
 * Styled main content area with ML operation optimizations
 */
const MainContent = styled('main')(({ theme }) => ({
  flex: 1,
  width: '100%',
  paddingTop: theme.spacing(8),
  paddingBottom: theme.spacing(4),
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  position: 'relative',
  zIndex: 1,
  [theme.breakpoints.up('lg')]: {
    paddingTop: theme.spacing(10),
  },
  [theme.breakpoints.up('uhd')]: {
    paddingTop: theme.spacing(12),
  },
  // ML-specific optimizations
  '& .ml-container': {
    width: '100%',
    maxWidth: theme.breakpoints.values.lg,
    margin: '0 auto',
    padding: theme.spacing(0, 2),
    [theme.breakpoints.up('xl')]: {
      maxWidth: theme.breakpoints.values.xl,
    },
  },
  // WebGL canvas container
  '& .webgl-container': {
    position: 'relative',
    width: '100%',
    height: '100%',
    minHeight: 400,
    backgroundColor: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    overflow: 'hidden',
  },
}));

/**
 * Layout component providing application structure with ML optimizations
 * Implements theme switching and responsive design
 */
const Layout: React.FC<LayoutProps> = React.memo(({ children, version }) => {
  // Theme preference detection
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const [isDarkMode, setIsDarkMode] = React.useState(prefersDarkMode);

  // Theme change handler with smooth transition
  const handleThemeToggle = React.useCallback((isDark: boolean) => {
    setIsDarkMode(isDark);
  }, []);

  // Current theme based on mode
  const currentTheme = React.useMemo(
    () => isDarkMode ? darkTheme : lightTheme,
    [isDarkMode]
  );

  // Performance monitoring for ML operations
  React.useEffect(() => {
    const reportWebGLMemory = () => {
      const canvas = document.querySelector('canvas');
      if (canvas) {
        const gl = canvas.getContext('webgl2');
        if (gl) {
          const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
          if (debugInfo) {
            console.debug('GPU:', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
          }
        }
      }
    };

    // Monitor WebGL context periodically
    const intervalId = setInterval(reportWebGLMemory, 5000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <ThemeProvider theme={currentTheme}>
      <CssBaseline />
      <LayoutContainer>
        <Header
          title="Video Game Diffusion Model"
          onThemeToggle={handleThemeToggle}
          isDarkMode={isDarkMode}
          ariaLabel={`Switch to ${isDarkMode ? 'light' : 'dark'} theme`}
        />
        
        <MainContent role="main" aria-label="Main content">
          <div className="ml-container">
            {children}
          </div>
        </MainContent>

        <Footer version={version} />
      </LayoutContainer>
    </ThemeProvider>
  );
});

// Display name for debugging
Layout.displayName = 'Layout';

export default Layout;