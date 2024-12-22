/**
 * @fileoverview WebGL context management for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { DEFAULT_WEBGL_CONFIG } from '../../config/webgl';
import { Logger } from '../utils/logger';

/**
 * Interface for WebGL context state preservation
 */
interface WebGLContextState {
    viewport: number[];
    scissor: number[];
    blendFunc: number[];
    clearColor: number[];
    program: WebGLProgram | null;
    framebuffer: WebGLFramebuffer | null;
}

/**
 * Manages WebGL context lifecycle and configuration for GPU-accelerated tensor operations
 */
export class WebGLContextManager {
    private gl: WebGLRenderingContext | null;
    private canvas: HTMLCanvasElement;
    private logger: Logger;
    private isContextLost: boolean;
    private contextAttributes: WebGLContextAttributes;
    private resources: Map<string, WebGLObject>;
    private savedState: WebGLContextState | null;
    private lastPerformanceCheck: number;
    private performanceMonitorInterval: number;
    private disposalPending: boolean;

    constructor(
        config: typeof DEFAULT_WEBGL_CONFIG,
        existingCanvas?: HTMLCanvasElement
    ) {
        this.logger = new Logger({
            level: 'info',
            namespace: 'webgl-context',
            persistLogs: true
        });

        this.canvas = existingCanvas || document.createElement('canvas');
        this.contextAttributes = {
            alpha: config.alpha,
            antialias: config.antialias,
            depth: config.depth,
            failIfMajorPerformanceCaveat: config.failIfMajorPerformanceCaveat,
            premultipliedAlpha: config.premultipliedAlpha,
            preserveDrawingBuffer: config.preserveDrawingBuffer,
            stencil: config.stencil,
            desynchronized: config.desynchronized,
            powerPreference: config.powerPreference
        };

        this.resources = new Map();
        this.isContextLost = false;
        this.savedState = null;
        this.lastPerformanceCheck = Date.now();
        this.performanceMonitorInterval = config.performanceFlags.optimizeMemoryUsage ? 1000 : 5000;
        this.disposalPending = false;

        this.initializeContext();
        this.setupContextLossHandling();
    }

    /**
     * Initializes WebGL2 context with optimal configuration
     */
    private initializeContext(): void {
        try {
            this.gl = this.canvas.getContext('webgl2', this.contextAttributes) as WebGLRenderingContext;

            if (!this.gl) {
                throw new Error('WebGL 2.0 initialization failed');
            }

            this.configureContext();
            this.logger.log('WebGL2 context initialized successfully', 'info', {
                vendor: this.gl.getParameter(this.gl.VENDOR),
                renderer: this.gl.getParameter(this.gl.RENDERER),
                version: this.gl.getParameter(this.gl.VERSION)
            });

        } catch (error) {
            this.logger.error(error as Error, { component: 'WebGLContextManager' });
            throw error;
        }
    }

    /**
     * Configures initial WebGL context state
     */
    private configureContext(): void {
        if (!this.gl) return;

        // Enable required extensions
        const requiredExtensions = ['EXT_color_buffer_float', 'OES_texture_float_linear'];
        for (const ext of requiredExtensions) {
            if (!this.gl.getExtension(ext)) {
                this.logger.warn(`Extension ${ext} not available`, { component: 'WebGLContextManager' });
            }
        }

        // Configure viewport and initial state
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0, 0, 0, 1);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    }

    /**
     * Sets up context loss and restoration handling
     */
    private setupContextLossHandling(): void {
        this.canvas.addEventListener('webglcontextlost', this.handleContextLoss.bind(this), false);
        this.canvas.addEventListener('webglcontextrestored', this.handleContextRestored.bind(this), false);
    }

    /**
     * Handles WebGL context loss with state preservation
     */
    private handleContextLoss(event: WebGLContextEvent): void {
        event.preventDefault();
        this.isContextLost = true;
        this.saveContextState();
        this.logger.warn('WebGL context lost', { 
            timestamp: Date.now(),
            reason: event.statusMessage
        });

        // Notify TensorFlow.js of context loss
        tf.engine().endScope();
        tf.engine().startScope();
    }

    /**
     * Handles WebGL context restoration
     */
    private handleContextRestored(): void {
        this.isContextLost = false;
        this.initializeContext();
        this.restoreContextState();
        this.logger.log('WebGL context restored', 'info', {
            timestamp: Date.now()
        });
    }

    /**
     * Saves current WebGL context state
     */
    private saveContextState(): void {
        if (!this.gl) return;

        this.savedState = {
            viewport: this.gl.getParameter(this.gl.VIEWPORT),
            scissor: this.gl.getParameter(this.gl.SCISSOR_BOX),
            blendFunc: [
                this.gl.getParameter(this.gl.BLEND_SRC_RGB),
                this.gl.getParameter(this.gl.BLEND_DST_RGB)
            ],
            clearColor: this.gl.getParameter(this.gl.COLOR_CLEAR_VALUE),
            program: this.gl.getParameter(this.gl.CURRENT_PROGRAM),
            framebuffer: this.gl.getParameter(this.gl.FRAMEBUFFER_BINDING)
        };
    }

    /**
     * Restores saved WebGL context state
     */
    private restoreContextState(): void {
        if (!this.gl || !this.savedState) return;

        this.gl.viewport(...this.savedState.viewport);
        this.gl.scissor(...this.savedState.scissor);
        this.gl.blendFunc(...this.savedState.blendFunc);
        this.gl.clearColor(...this.savedState.clearColor);
        this.gl.useProgram(this.savedState.program);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.savedState.framebuffer);
    }

    /**
     * Checks WebGL performance and memory usage
     */
    public checkPerformance(): void {
        if (!this.gl || Date.now() - this.lastPerformanceCheck < this.performanceMonitorInterval) {
            return;
        }

        this.lastPerformanceCheck = Date.now();
        const memoryInfo = (this.gl as any).getExtension('WEBGL_debug_renderer_info');
        
        if (memoryInfo) {
            this.logger.logPerformance('webgl_memory', {
                vendor: this.gl.getParameter(memoryInfo.UNMASKED_VENDOR_WEBGL),
                renderer: this.gl.getParameter(memoryInfo.UNMASKED_RENDERER_WEBGL),
                totalResources: this.resources.size
            });
        }
    }

    /**
     * Disposes WebGL context and resources
     */
    public dispose(): void {
        if (this.disposalPending) return;
        this.disposalPending = true;

        try {
            // Clean up tracked resources
            this.resources.forEach((resource) => {
                if (resource instanceof WebGLBuffer) {
                    this.gl?.deleteBuffer(resource);
                } else if (resource instanceof WebGLTexture) {
                    this.gl?.deleteTexture(resource);
                } else if (resource instanceof WebGLFramebuffer) {
                    this.gl?.deleteFramebuffer(resource);
                }
            });
            this.resources.clear();

            // Remove event listeners
            this.canvas.removeEventListener('webglcontextlost', this.handleContextLoss);
            this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored);

            // Lose context explicitly
            const ext = this.gl?.getExtension('WEBGL_lose_context');
            if (ext) {
                ext.loseContext();
            }

            this.gl = null;
            this.savedState = null;
            this.logger.log('WebGL context disposed', 'info');

        } catch (error) {
            this.logger.error(error as Error, { component: 'WebGLContextManager', operation: 'dispose' });
        } finally {
            this.disposalPending = false;
        }
    }

    /**
     * Gets current WebGL context
     */
    public getContext(): WebGLRenderingContext | null {
        return this.gl;
    }

    /**
     * Gets canvas element
     */
    public getCanvas(): HTMLCanvasElement {
        return this.canvas;
    }
}