/**
 * Entry point for browser-based video game diffusion model React application
 * Implements WebGL-accelerated TensorFlow.js initialization and memory monitoring
 * @version 1.0.0
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as tf from '@tensorflow/tfjs';

// Internal imports
import App from './App';
import { lightTheme } from './styles/theme';
import './styles/global.css';
import { validateBrowserCapabilities } from './lib/utils/validation';
import { Logger } from './lib/utils/logger';
import { MEMORY_CONSTRAINTS, PERFORMANCE_THRESHOLDS } from './constants/model';

// Initialize logger
const logger = new Logger({
  level: 'info',
  namespace: 'bvgdm-init',
  persistLogs: true,
  metricsRetentionMs: 3600000
});

/**
 * Initializes TensorFlow.js with WebGL backend and memory optimization
 */
async function initializeTensorFlow(): Promise<void> {
  try {
    // Set backend to WebGL and wait for initialization
    await tf.setBackend('webgl');
    await tf.ready();

    // Configure WebGL for optimal performance
    tf.env().set('WEBGL_VERSION', 2);
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
    tf.env().set('WEBGL_PACK', true);
    tf.env().set('WEBGL_CPU_FORWARD', false);
    tf.env().set('WEBGL_MAX_TEXTURE_SIZE', 4096);

    // Configure memory management
    tf.engine().startScope();
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 
      MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD * MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE);

    logger.info('TensorFlow.js initialized successfully', {
      backend: tf.getBackend(),
      webglVersion: tf.env().get('WEBGL_VERSION'),
      maxTextureSize: tf.env().get('WEBGL_MAX_TEXTURE_SIZE')
    });

  } catch (error) {
    logger.error('TensorFlow.js initialization failed', error);
    throw error;
  }
}

/**
 * Validates browser compatibility and required features
 */
async function checkBrowserCompatibility(): Promise<boolean> {
  try {
    const result = await validateBrowserCapabilities({
      webgl2Required: true,
      checkWebWorker: true
    });

    if (!result.isCompatible) {
      logger.error('Browser compatibility check failed', {
        missingFeatures: result.missingFeatures,
        webglInfo: result.webglInfo
      });
      return false;
    }

    logger.info('Browser compatibility check passed', {
      webglInfo: result.webglInfo
    });
    return true;

  } catch (error) {
    logger.error('Browser compatibility check failed', error);
    return false;
  }
}

/**
 * Sets up memory monitoring and cleanup
 */
function setupMemoryMonitoring(): void {
  const memoryCheckInterval = setInterval(async () => {
    const memoryInfo = await tf.memory();
    const memoryUsage = memoryInfo.numBytes / (1024 * 1024); // Convert to MB

    logger.logPerformance('memory_usage', {
      totalBytes: memoryInfo.numBytes,
      numTensors: memoryInfo.numTensors,
      numDataBuffers: memoryInfo.numDataBuffers,
      unreliable: memoryInfo.unreliable
    });

    // Trigger cleanup if memory usage exceeds threshold
    if (memoryUsage > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.9) {
      logger.warn('High memory usage detected, triggering cleanup');
      tf.engine().endScope();
      tf.engine().startScope();
    }
  }, PERFORMANCE_THRESHOLDS.MEMORY_ALERT_THRESHOLD);

  // Cleanup on window unload
  window.addEventListener('unload', () => {
    clearInterval(memoryCheckInterval);
    tf.engine().endScope();
  });
}

/**
 * Initializes and renders the application
 */
async function renderApp(): Promise<void> {
  const rootElement = document.getElementById('root');
  if (!rootElement) {
    throw new Error('Root element not found');
  }

  try {
    // Check browser compatibility
    const isCompatible = await checkBrowserCompatibility();
    if (!isCompatible) {
      throw new Error('Browser compatibility check failed');
    }

    // Initialize TensorFlow.js
    await initializeTensorFlow();

    // Set up memory monitoring
    setupMemoryMonitoring();

    // Create root and render app
    const root = ReactDOM.createRoot(rootElement);
    root.render(
      <React.StrictMode>
        <ThemeProvider theme={lightTheme}>
          <CssBaseline />
          <App />
        </ThemeProvider>
      </React.StrictMode>
    );

    // Remove loading indicator
    const loadingElement = document.getElementById('loading');
    if (loadingElement?.parentNode) {
      loadingElement.parentNode.removeChild(loadingElement);
    }

    logger.info('Application rendered successfully');

  } catch (error) {
    logger.error('Application initialization failed', error);
    // Display error message to user
    rootElement.innerHTML = `
      <div style="padding: 20px; text-align: center;">
        <h1>Application Error</h1>
        <p>Failed to initialize application. Please ensure your browser supports WebGL 2.0.</p>
      </div>
    `;
  }
}

// Initialize application
renderApp().catch(error => {
  logger.error('Fatal application error', error);
});