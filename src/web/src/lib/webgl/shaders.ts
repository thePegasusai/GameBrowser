/**
 * @fileoverview Enhanced WebGL shader management for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import '@tensorflow/tfjs-backend-webgl'; // v4.x
import { WebGLContextManager } from './context';
import { TensorSpec } from '../../types/tensor';
import { Logger } from '../utils/logger';

/**
 * Interface for WebGL shader program information with performance tracking
 */
interface ShaderProgramInfo {
    id: string;
    program: WebGLProgram;
    vertexShaderId: string;
    fragmentShaderId: string;
    isValid: boolean;
    compilationTime: number;
    lastExecutionTime: number;
    requiredFeatures: WebGLFeatures;
}

/**
 * Interface for WebGL feature requirements
 */
interface WebGLFeatures {
    floatTextures: boolean;
    linearFiltering: boolean;
    vertexArrayObjects: boolean;
    instancedArrays: boolean;
}

/**
 * Interface for shader compilation options
 */
interface ShaderOptions {
    debug?: boolean;
    optimize?: boolean;
    timeout?: number;
}

/**
 * Enhanced WebGL shader manager with performance monitoring and error handling
 */
export class WebGLShaderManager {
    private readonly gl: WebGLRenderingContext;
    private readonly programs: Map<string, WebGLProgram>;
    private readonly vertexShaders: Map<string, WebGLShader>;
    private readonly fragmentShaders: Map<string, WebGLShader>;
    private readonly programInfo: Map<string, ShaderProgramInfo>;
    private readonly logger: Logger;
    private lastPerformanceCheck: number;
    private readonly performanceInterval: number = 1000;

    constructor(contextManager: WebGLContextManager) {
        this.gl = contextManager.getContext()!;
        if (!this.gl) {
            throw new Error('WebGL context initialization failed');
        }

        this.programs = new Map();
        this.vertexShaders = new Map();
        this.fragmentShaders = new Map();
        this.programInfo = new Map();
        this.lastPerformanceCheck = Date.now();

        this.logger = new Logger({
            level: 'info',
            namespace: 'webgl-shaders',
            persistLogs: true
        });

        this.validateWebGLFeatures();
    }

    /**
     * Creates and compiles a WebGL shader with enhanced validation
     */
    public createShader(
        source: string,
        type: number,
        options: ShaderOptions = {}
    ): WebGLShader {
        const startTime = performance.now();
        const shader = this.gl.createShader(type);

        if (!shader) {
            throw new Error('Failed to create shader');
        }

        try {
            // Set shader source and compile
            this.gl.shaderSource(shader, source);
            this.gl.compileShader(shader);

            // Validate compilation
            if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
                const error = this.gl.getShaderInfoLog(shader);
                this.gl.deleteShader(shader);
                throw new Error(`Shader compilation failed: ${error}`);
            }

            // Store in appropriate collection
            const shaderId = this.generateShaderId(type, source);
            if (type === this.gl.VERTEX_SHADER) {
                this.vertexShaders.set(shaderId, shader);
            } else {
                this.fragmentShaders.set(shaderId, shader);
            }

            // Performance tracking
            const compilationTime = performance.now() - startTime;
            this.logger.recordMetric('shader_compilation_time', compilationTime, 'performance', {
                type: type === this.gl.VERTEX_SHADER ? 'vertex' : 'fragment'
            });

            return shader;

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'WebGLShaderManager',
                operation: 'createShader',
                shaderType: type === this.gl.VERTEX_SHADER ? 'vertex' : 'fragment'
            });
            throw error;
        }
    }

    /**
     * Creates and links a WebGL program with performance monitoring
     */
    public createProgram(
        vertexShader: WebGLShader,
        fragmentShader: WebGLShader,
        options: ShaderOptions = {}
    ): WebGLProgram {
        const startTime = performance.now();
        const program = this.gl.createProgram();

        if (!program) {
            throw new Error('Failed to create shader program');
        }

        try {
            // Attach shaders and link program
            this.gl.attachShader(program, vertexShader);
            this.gl.attachShader(program, fragmentShader);
            this.gl.linkProgram(program);

            // Validate linking
            if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
                const error = this.gl.getProgramInfoLog(program);
                this.gl.deleteProgram(program);
                throw new Error(`Program linking failed: ${error}`);
            }

            // Store program information
            const programId = this.generateProgramId(vertexShader, fragmentShader);
            this.programs.set(programId, program);
            this.programInfo.set(programId, {
                id: programId,
                program,
                vertexShaderId: this.getShaderId(vertexShader),
                fragmentShaderId: this.getShaderId(fragmentShader),
                isValid: true,
                compilationTime: performance.now() - startTime,
                lastExecutionTime: 0,
                requiredFeatures: this.getRequiredFeatures(program)
            });

            return program;

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'WebGLShaderManager',
                operation: 'createProgram'
            });
            throw error;
        }
    }

    /**
     * Uses a shader program with performance tracking
     */
    public useProgram(program: WebGLProgram): void {
        const startTime = performance.now();
        try {
            this.gl.useProgram(program);
            
            // Update performance metrics
            const programId = this.getProgramId(program);
            if (programId) {
                const info = this.programInfo.get(programId);
                if (info) {
                    info.lastExecutionTime = performance.now() - startTime;
                }
            }

            this.checkPerformance();

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'WebGLShaderManager',
                operation: 'useProgram'
            });
            throw error;
        }
    }

    /**
     * Deletes a shader program and its resources
     */
    public deleteProgram(program: WebGLProgram): void {
        try {
            const programId = this.getProgramId(program);
            if (programId) {
                const info = this.programInfo.get(programId);
                if (info) {
                    // Delete associated shaders
                    const vertexShader = this.vertexShaders.get(info.vertexShaderId);
                    const fragmentShader = this.fragmentShaders.get(info.fragmentShaderId);

                    if (vertexShader) {
                        this.gl.deleteShader(vertexShader);
                        this.vertexShaders.delete(info.vertexShaderId);
                    }
                    if (fragmentShader) {
                        this.gl.deleteShader(fragmentShader);
                        this.fragmentShaders.delete(info.fragmentShaderId);
                    }

                    // Delete program
                    this.gl.deleteProgram(program);
                    this.programs.delete(programId);
                    this.programInfo.delete(programId);
                }
            }
        } catch (error) {
            this.logger.error(error as Error, {
                component: 'WebGLShaderManager',
                operation: 'deleteProgram'
            });
        }
    }

    /**
     * Gets performance metrics for shader operations
     */
    public getPerformanceMetrics(): Record<string, any> {
        const metrics: Record<string, any> = {
            programs: this.programs.size,
            vertexShaders: this.vertexShaders.size,
            fragmentShaders: this.fragmentShaders.size,
            averageCompilationTime: 0,
            averageExecutionTime: 0
        };

        let totalCompilationTime = 0;
        let totalExecutionTime = 0;
        let count = 0;

        this.programInfo.forEach(info => {
            totalCompilationTime += info.compilationTime;
            totalExecutionTime += info.lastExecutionTime;
            count++;
        });

        if (count > 0) {
            metrics.averageCompilationTime = totalCompilationTime / count;
            metrics.averageExecutionTime = totalExecutionTime / count;
        }

        return metrics;
    }

    // Private helper methods
    private validateWebGLFeatures(): void {
        const requiredExtensions = [
            'OES_texture_float',
            'EXT_color_buffer_float',
            'OES_texture_float_linear'
        ];

        for (const ext of requiredExtensions) {
            if (!this.gl.getExtension(ext)) {
                this.logger.warn(`Required WebGL extension ${ext} not available`, {
                    component: 'WebGLShaderManager'
                });
            }
        }
    }

    private generateShaderId(type: number, source: string): string {
        return `${type}-${tf.util.hashCode(source)}`;
    }

    private generateProgramId(vertexShader: WebGLShader, fragmentShader: WebGLShader): string {
        return `program-${this.getShaderId(vertexShader)}-${this.getShaderId(fragmentShader)}`;
    }

    private getShaderId(shader: WebGLShader): string {
        for (const [id, s] of this.vertexShaders) {
            if (s === shader) return id;
        }
        for (const [id, s] of this.fragmentShaders) {
            if (s === shader) return id;
        }
        throw new Error('Shader not found');
    }

    private getProgramId(program: WebGLProgram): string | undefined {
        for (const [id, p] of this.programs) {
            if (p === program) return id;
        }
        return undefined;
    }

    private getRequiredFeatures(program: WebGLProgram): WebGLFeatures {
        return {
            floatTextures: true,
            linearFiltering: true,
            vertexArrayObjects: this.gl.getExtension('OES_vertex_array_object') !== null,
            instancedArrays: this.gl.getExtension('ANGLE_instanced_arrays') !== null
        };
    }

    private checkPerformance(): void {
        const now = Date.now();
        if (now - this.lastPerformanceCheck >= this.performanceInterval) {
            const metrics = this.getPerformanceMetrics();
            this.logger.recordMetric('shader_performance', metrics.averageExecutionTime, 'performance');
            this.lastPerformanceCheck = now;
        }
    }
}