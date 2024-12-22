/**
 * @fileoverview Central export point for all application constants
 * Aggregates model, UI, and system-wide configuration values
 * @version 1.0.0
 * @license MIT
 */

// Import model-related constants
import {
  MODEL_ARCHITECTURES,
  VAE_ARCHITECTURES,
  MEMORY_CONSTRAINTS,
  PERFORMANCE_THRESHOLDS,
  BROWSER_COMPATIBILITY,
  DEFAULT_TENSOR_SPECS,
  DEFAULT_DIT_CONFIG,
  DEFAULT_VAE_CONFIG,
  TRAINING_PARAMS,
  VALIDATION_INTERVALS
} from './model';

// Import UI-related constants
import {
  THEME_COLORS,
  LAYOUT_SIZES,
  COMPONENT_SIZES,
  ANIMATION_TIMINGS,
  VIDEO_DISPLAY,
  INTERACTION_LIMITS
} from './ui';

// Import type definitions
import { ModelConfig, BrowserFeature, MemoryStatus, BrowserStatus } from '../types/model';

/**
 * System-wide performance constraints
 * Enforces <50ms inference and <4GB memory usage
 */
export const SYSTEM_CONSTRAINTS = {
  /** Maximum memory usage in GB */
  MAX_MEMORY_GB: 4,
  /** Maximum frame processing time in ms */
  MAX_FRAME_TIME_MS: 50,
  /** Minimum frames per second */
  MIN_FPS: 30,
  /** Memory cleanup threshold percentage */
  MEMORY_CLEANUP_THRESHOLD: 0.85,
  /** Performance check interval in ms */
  PERFORMANCE_CHECK_INTERVAL: 1000
} as const;

/**
 * Browser compatibility requirements
 * Specifies minimum versions and required features
 */
export const BROWSER_REQUIREMENTS = {
  /** Minimum browser versions */
  VERSIONS: {
    CHROME: 90,
    FIREFOX: 88,
    SAFARI: 14,
    EDGE: 90
  },
  /** Required browser features */
  FEATURES: [
    BrowserFeature.WEBGL2,
    BrowserFeature.WEBWORKER,
    BrowserFeature.INDEXEDDB,
    BrowserFeature.SHAREDARRAYBUFFER
  ],
  /** WebGL requirements */
  WEBGL: {
    VERSION: 2.0,
    MIN_TEXTURE_SIZE: 2048
  }
} as const;

/**
 * Responsive design breakpoints
 * Supports layouts from 720p to 4K
 */
export const RESPONSIVE_BREAKPOINTS = {
  /** Minimum supported dimensions */
  MIN_WIDTH: 1280,
  MIN_HEIGHT: 720,
  /** Maximum supported dimensions */
  MAX_WIDTH: 3840,
  MAX_HEIGHT: 2160,
  /** Standard aspect ratio */
  ASPECT_RATIO: 16 / 9
} as const;

/**
 * Validates system constraints for memory and performance
 * @param constraints System constraints to validate
 * @returns Promise<boolean> indicating if system meets requirements
 */
export async function validateSystemConstraints(
  constraints = SYSTEM_CONSTRAINTS
): Promise<boolean> {
  try {
    // Check available memory
    const memory = await navigator.deviceMemory;
    if (memory < constraints.MAX_MEMORY_GB) {
      console.warn(`Insufficient memory: ${memory}GB < ${constraints.MAX_MEMORY_GB}GB`);
      return false;
    }

    // Verify WebGL support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      console.warn('WebGL 2 not supported');
      return false;
    }

    // Check browser version
    const userAgent = navigator.userAgent.toLowerCase();
    const browserVersion = parseInt(userAgent.match(/(?:chrome|firefox|safari|edge)\/(\d+)/)?.[1] || '0');
    const minVersion = BROWSER_REQUIREMENTS.VERSIONS[
      Object.keys(BROWSER_REQUIREMENTS.VERSIONS).find(
        browser => userAgent.includes(browser.toLowerCase())
      ) as keyof typeof BROWSER_REQUIREMENTS.VERSIONS
    ];
    
    if (browserVersion < minVersion) {
      console.warn(`Unsupported browser version: ${browserVersion} < ${minVersion}`);
      return false;
    }

    return true;
  } catch (error) {
    console.error('Error validating system constraints:', error);
    return false;
  }
}

// Re-export all constants
export {
  // Model-related exports
  MODEL_ARCHITECTURES,
  VAE_ARCHITECTURES,
  MEMORY_CONSTRAINTS,
  PERFORMANCE_THRESHOLDS,
  BROWSER_COMPATIBILITY,
  DEFAULT_TENSOR_SPECS,
  DEFAULT_DIT_CONFIG,
  DEFAULT_VAE_CONFIG,
  TRAINING_PARAMS,
  VALIDATION_INTERVALS,
  
  // UI-related exports
  THEME_COLORS,
  LAYOUT_SIZES,
  COMPONENT_SIZES,
  ANIMATION_TIMINGS,
  VIDEO_DISPLAY,
  INTERACTION_LIMITS
};