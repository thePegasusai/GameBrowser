/**
 * UI Constants for Browser-based Video Game Diffusion Model
 * Follows Material Design principles and supports responsive layouts
 * @version 1.0.0
 */

/**
 * Core theme colors following Material Design color system
 * Used for consistent visual hierarchy across the application
 */
export const THEME_COLORS = {
  /** Primary brand color - Material Blue 700 */
  PRIMARY: '#1976d2',
  /** Secondary UI color - Material Grey 800 */
  SECONDARY: '#424242',
  /** Error state color - Material Red 700 */
  ERROR: '#d32f2f',
  /** Success state color - Material Green 700 */
  SUCCESS: '#388e3c',
  /** Warning state color - Material Orange 700 */
  WARNING: '#f57c00'
} as const;

/**
 * Layout dimensions for major UI sections
 * Supports responsive design from 1280x720 to 4K resolutions
 */
export const LAYOUT_SIZES = {
  /** Standard Material Design app bar height */
  HEADER_HEIGHT: 64,
  /** Standard Material Design navigation drawer width */
  SIDEBAR_WIDTH: 240,
  /** Maximum content width for optimal readability */
  CONTENT_MAX_WIDTH: 1440,
  /** Standard footer height */
  FOOTER_HEIGHT: 48
} as const;

/**
 * Standard sizes for common UI components
 * Following Material Design metrics for consistent component sizing
 */
export const COMPONENT_SIZES = {
  /** Standard button height for normal emphasis buttons */
  BUTTON_HEIGHT: 36,
  /** Standard height for text input fields */
  INPUT_HEIGHT: 40,
  /** Standard height for slider track */
  SLIDER_HEIGHT: 4,
  /** Standard height for progress indicators */
  PROGRESS_HEIGHT: 4
} as const;

/**
 * Animation durations for UI transitions
 * Following Material Design motion guidelines for natural movement
 */
export const ANIMATION_TIMINGS = {
  /** Duration for micro-interactions (150ms) */
  FAST: 150,
  /** Duration for standard transitions (300ms) */
  NORMAL: 300,
  /** Duration for complex animations (500ms) */
  SLOW: 500
} as const;

/**
 * Video display dimensions and constraints
 * Supporting responsive layouts while maintaining aspect ratio
 */
export const VIDEO_DISPLAY = {
  /** Minimum supported video width (720p) */
  MIN_WIDTH: 1280,
  /** Maximum supported video width (4K) */
  MAX_WIDTH: 3840,
  /** 16:9 aspect ratio (1.7778) */
  ASPECT_RATIO: 1.7778,
  /** Preview thumbnail size */
  PREVIEW_SIZE: 256
} as const;

/**
 * Interaction thresholds and limits
 * Optimizing user experience through carefully tuned interaction parameters
 */
export const INTERACTION_LIMITS = {
  /** Minimum distance in pixels to trigger drag operations */
  MIN_DRAG_DISTANCE: 10,
  /** Maximum delay in ms to detect double-click */
  DOUBLE_CLICK_DELAY: 300,
  /** Scroll distance in pixels before triggering scroll actions */
  SCROLL_THRESHOLD: 50
} as const;