/**
 * Global test setup configuration for browser-based video game diffusion model
 * Version: 1.0.0
 * 
 * Configures Jest environment with:
 * - DOM testing utilities (@testing-library/jest-dom@5.x)
 * - Canvas/WebGL mocks (jest-canvas-mock@2.x)
 * - TensorFlow.js testing infrastructure (@tensorflow/tfjs-core@4.x)
 * - WebGL backend mocking (@tensorflow/tfjs-backend-webgl@4.x)
 */

import '@testing-library/jest-dom';
import 'jest-canvas-mock';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

// Mock TensorFlow.js core functionality
jest.mock('@tensorflow/tfjs-core', () => ({
  ...jest.requireActual('@tensorflow/tfjs-core'),
  setBackend: jest.fn(),
  ready: jest.fn().mockResolvedValue(true),
  memory: jest.fn().mockReturnValue({
    numBytes: 0,
    numTensors: 0,
    numDataBuffers: 0,
    unreliable: false
  })
}));

// Mock TensorFlow.js WebGL backend
jest.mock('@tensorflow/tfjs-backend-webgl', () => ({
  ...jest.requireActual('@tensorflow/tfjs-backend-webgl'),
  MathBackendWebGL: jest.fn().mockImplementation(() => ({
    dispose: jest.fn(),
    memory: jest.fn(),
    getGPGPUContext: jest.fn()
  }))
}));

/**
 * Sets up global mocks required for testing browser environment and ML operations
 */
export function setupGlobalMocks(): void {
  // Mock ResizeObserver
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  // Mock text encoding/decoding
  global.TextEncoder = TextEncoder;
  global.TextDecoder = TextDecoder;

  // Mock WebGL contexts
  global.WebGLRenderingContext = class WebGLRenderingContext {
    canvas: HTMLCanvasElement;
    drawingBufferWidth: number = 0;
    drawingBufferHeight: number = 0;

    constructor() {
      this.canvas = document.createElement('canvas');
    }

    getExtension(name: string) {
      return {};
    }

    getParameter(parameter: number) {
      return 0;
    }
  } as any;

  global.WebGL2RenderingContext = class WebGL2RenderingContext extends (global.WebGLRenderingContext as any) {} as any;

  // Mock video element
  global.HTMLVideoElement = class HTMLVideoElement extends HTMLElement {
    width: number = 0;
    height: number = 0;
    videoWidth: number = 0;
    videoHeight: number = 0;
    currentTime: number = 0;
    duration: number = 0;
    paused: boolean = true;
    ended: boolean = false;

    play() { return Promise.resolve(); }
    pause() {}
  } as any;

  // Mock ImageData
  global.ImageData = class ImageData {
    width: number;
    height: number;
    data: Uint8ClampedArray;

    constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  };
}

/**
 * Configures TensorFlow.js backend for testing with memory management and WebGL support
 */
export function setupTensorFlowBackend(): void {
  const mockTensorOps = {
    dispose: jest.fn(),
    dataSync: jest.fn().mockReturnValue(new Float32Array(1)),
    buffer: jest.fn().mockReturnValue({
      toTensor: jest.fn()
    })
  };

  // Mock tensor operations
  jest.spyOn(tf, 'tensor').mockImplementation(() => mockTensorOps as any);
  jest.spyOn(tf, 'tensor2d').mockImplementation(() => mockTensorOps as any);
  jest.spyOn(tf, 'tensor3d').mockImplementation(() => mockTensorOps as any);
  jest.spyOn(tf, 'tensor4d').mockImplementation(() => mockTensorOps as any);

  // Configure WebGL context
  const mockWebGLContext = {
    createTexture: jest.fn(),
    bindTexture: jest.fn(),
    texImage2D: jest.fn(),
    texParameteri: jest.fn(),
    getExtension: jest.fn().mockReturnValue({}),
    getParameter: jest.fn().mockReturnValue(4096) // MAX_TEXTURE_SIZE
  };

  // Mock WebGL context creation
  jest.spyOn(HTMLCanvasElement.prototype, 'getContext')
    .mockImplementation(() => mockWebGLContext as any);
}

/**
 * Configures video element and processing mocks for testing
 */
export function setupVideoMocks(): void {
  const mockVideoElement = {
    play: jest.fn().mockResolvedValue(undefined),
    pause: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    width: 640,
    height: 480,
    videoWidth: 640,
    videoHeight: 480,
    currentTime: 0,
    duration: 60,
    readyState: 4, // HAVE_ENOUGH_DATA
    HAVE_ENOUGH_DATA: 4
  };

  // Mock video element creation
  jest.spyOn(document, 'createElement')
    .mockImplementation((tagName: string) => {
      if (tagName.toLowerCase() === 'video') {
        return mockVideoElement as any;
      }
      return document.createElement(tagName);
    });

  // Mock requestAnimationFrame
  global.requestAnimationFrame = jest.fn().mockImplementation(cb => setTimeout(cb, 0));
  global.cancelAnimationFrame = jest.fn().mockImplementation(id => clearTimeout(id));
}

// Initialize global test environment
setupGlobalMocks();
setupTensorFlowBackend();
setupVideoMocks();