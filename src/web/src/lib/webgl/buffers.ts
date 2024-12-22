/**
 * @fileoverview Enhanced WebGL buffer management for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { TensorMemoryManager } from '../tensor/memory';
import { TensorSpec } from '../../types/tensor';
import { WebGLContextManager } from './context';
import { Logger } from '../utils/logger';

/**
 * Interface for enhanced buffer information tracking
 */
interface BufferInfo {
    byteSize: number;
    lastUsed: number;
    isValid: boolean;
    target: number;
    usageCount: number;
    averageBindTime: number;
    isPriority: boolean;
    performanceMetrics: {
        createTime: number;
        totalBindTime: number;
        lastBindDuration: number;
    };
}

/**
 * Configuration for buffer pooling and optimization
 */
interface BufferPoolConfig {
    maxPoolSize: number;
    reuseThreshold: number;
    cleanupInterval: number;
    prioritySettings: {
        minUsageCount: number;
        ageThreshold: number;
    };
}

/**
 * Enhanced WebGL buffer manager with advanced memory optimization and monitoring
 */
export class WebGLBufferManager {
    private gl: WebGLRenderingContext;
    private buffers: Map<string, WebGLBuffer>;
    private bufferInfo: Map<string, BufferInfo>;
    private memoryManager: TensorMemoryManager;
    private contextManager: WebGLContextManager;
    private logger: Logger;
    private totalBufferSize: number;
    private lastCleanup: number;
    private poolConfig: BufferPoolConfig;
    private bufferPool: Map<number, WebGLBuffer[]>;

    constructor(
        memoryManager: TensorMemoryManager,
        contextManager: WebGLContextManager,
        poolConfig?: Partial<BufferPoolConfig>
    ) {
        this.memoryManager = memoryManager;
        this.contextManager = contextManager;
        this.gl = contextManager.getContext()!;
        this.logger = new Logger({
            level: 'info',
            namespace: 'webgl-buffer',
            persistLogs: true
        });

        this.buffers = new Map();
        this.bufferInfo = new Map();
        this.bufferPool = new Map();
        this.totalBufferSize = 0;
        this.lastCleanup = Date.now();

        // Initialize buffer pool configuration
        this.poolConfig = {
            maxPoolSize: poolConfig?.maxPoolSize || 100,
            reuseThreshold: poolConfig?.reuseThreshold || 1000,
            cleanupInterval: poolConfig?.cleanupInterval || 30000,
            prioritySettings: {
                minUsageCount: 5,
                ageThreshold: 60000
            }
        };

        // Set up automatic cleanup
        this.setupAutoCleanup();
    }

    /**
     * Creates or reuses a WebGL buffer with optimized memory management
     */
    public createBuffer(
        data: ArrayBuffer,
        target: number = this.gl.ARRAY_BUFFER,
        usage: number = this.gl.STATIC_DRAW,
        options: { isPriority?: boolean } = {}
    ): string {
        const startTime = performance.now();
        const bufferId = `buffer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        try {
            // Check memory pressure
            const memoryStatus = this.memoryManager.getMemoryInfo();
            if (memoryStatus.utilizationPercentage > 90) {
                this.cleanupUnusedBuffers();
            }

            // Try to reuse buffer from pool
            let buffer = this.reuseBufferFromPool(data.byteLength, target);
            let isReused = false;

            if (!buffer) {
                buffer = this.gl.createBuffer();
                if (!buffer) {
                    throw new Error('Failed to create WebGL buffer');
                }
            } else {
                isReused = true;
            }

            // Bind and initialize buffer
            this.gl.bindBuffer(target, buffer);
            this.gl.bufferData(target, data, usage);

            // Track buffer information
            this.buffers.set(bufferId, buffer);
            this.bufferInfo.set(bufferId, {
                byteSize: data.byteLength,
                lastUsed: Date.now(),
                isValid: true,
                target,
                usageCount: 0,
                averageBindTime: 0,
                isPriority: options.isPriority || false,
                performanceMetrics: {
                    createTime: performance.now() - startTime,
                    totalBindTime: 0,
                    lastBindDuration: 0
                }
            });

            this.totalBufferSize += data.byteLength;

            // Log buffer creation
            this.logger.log('Buffer created', 'debug', {
                bufferId,
                byteSize: data.byteLength,
                isReused,
                totalBufferSize: this.totalBufferSize
            });

            return bufferId;

        } catch (error) {
            this.logger.error(error as Error, {
                operation: 'createBuffer',
                dataSize: data.byteLength
            });
            throw error;
        }
    }

    /**
     * Binds a buffer with performance tracking
     */
    public bindBuffer(bufferId: string): void {
        const startTime = performance.now();
        const buffer = this.buffers.get(bufferId);
        const info = this.bufferInfo.get(bufferId);

        if (!buffer || !info || !info.isValid) {
            throw new Error(`Invalid buffer ID: ${bufferId}`);
        }

        try {
            this.gl.bindBuffer(info.target, buffer);
            
            // Update usage metrics
            info.lastUsed = Date.now();
            info.usageCount++;
            info.performanceMetrics.lastBindDuration = performance.now() - startTime;
            info.performanceMetrics.totalBindTime += info.performanceMetrics.lastBindDuration;
            info.averageBindTime = info.performanceMetrics.totalBindTime / info.usageCount;

        } catch (error) {
            this.logger.error(error as Error, {
                operation: 'bindBuffer',
                bufferId
            });
            throw error;
        }
    }

    /**
     * Updates buffer data with memory optimization
     */
    public updateBuffer(bufferId: string, data: ArrayBuffer, offset: number = 0): void {
        const buffer = this.buffers.get(bufferId);
        const info = this.bufferInfo.get(bufferId);

        if (!buffer || !info || !info.isValid) {
            throw new Error(`Invalid buffer ID: ${bufferId}`);
        }

        try {
            this.gl.bindBuffer(info.target, buffer);
            this.gl.bufferSubData(info.target, offset, data);

            // Update buffer info
            info.lastUsed = Date.now();
            info.byteSize = Math.max(info.byteSize, offset + data.byteLength);

        } catch (error) {
            this.logger.error(error as Error, {
                operation: 'updateBuffer',
                bufferId,
                dataSize: data.byteLength
            });
            throw error;
        }
    }

    /**
     * Deletes a buffer and manages memory cleanup
     */
    public deleteBuffer(bufferId: string): void {
        const buffer = this.buffers.get(bufferId);
        const info = this.bufferInfo.get(bufferId);

        if (!buffer || !info) return;

        try {
            if (info.isPriority && info.usageCount > this.poolConfig.prioritySettings.minUsageCount) {
                // Add to buffer pool for reuse
                this.addToBufferPool(buffer, info);
            } else {
                this.gl.deleteBuffer(buffer);
            }

            this.totalBufferSize -= info.byteSize;
            this.buffers.delete(bufferId);
            this.bufferInfo.delete(bufferId);

            this.logger.log('Buffer deleted', 'debug', {
                bufferId,
                byteSize: info.byteSize,
                totalBufferSize: this.totalBufferSize
            });

        } catch (error) {
            this.logger.error(error as Error, {
                operation: 'deleteBuffer',
                bufferId
            });
        }
    }

    /**
     * Gets buffer information and performance metrics
     */
    public getBufferInfo(bufferId: string): BufferInfo | null {
        return this.bufferInfo.get(bufferId) || null;
    }

    /**
     * Optimizes memory usage by cleaning up unused buffers
     */
    private cleanupUnusedBuffers(): void {
        const now = Date.now();
        const threshold = now - this.poolConfig.reuseThreshold;

        for (const [bufferId, info] of this.bufferInfo.entries()) {
            if (!info.isPriority && info.lastUsed < threshold) {
                this.deleteBuffer(bufferId);
            }
        }

        this.lastCleanup = now;
    }

    /**
     * Sets up automatic cleanup interval
     */
    private setupAutoCleanup(): void {
        setInterval(() => {
            if (Date.now() - this.lastCleanup >= this.poolConfig.cleanupInterval) {
                this.cleanupUnusedBuffers();
            }
        }, this.poolConfig.cleanupInterval);
    }

    /**
     * Attempts to reuse a buffer from the pool
     */
    private reuseBufferFromPool(byteSize: number, target: number): WebGLBuffer | null {
        const pooledBuffers = this.bufferPool.get(target) || [];
        const index = pooledBuffers.findIndex(buffer => {
            const info = Array.from(this.bufferInfo.values())
                .find(i => i.target === target && i.byteSize >= byteSize);
            return info !== undefined;
        });

        if (index !== -1) {
            const buffer = pooledBuffers[index];
            pooledBuffers.splice(index, 1);
            return buffer;
        }

        return null;
    }

    /**
     * Adds a buffer to the reuse pool
     */
    private addToBufferPool(buffer: WebGLBuffer, info: BufferInfo): void {
        const pooledBuffers = this.bufferPool.get(info.target) || [];
        if (pooledBuffers.length < this.poolConfig.maxPoolSize) {
            pooledBuffers.push(buffer);
            this.bufferPool.set(info.target, pooledBuffers);
        } else {
            this.gl.deleteBuffer(buffer);
        }
    }
}