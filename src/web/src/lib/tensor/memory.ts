/**
 * @fileoverview Advanced tensor memory management with WebGL optimization for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { dispose, memory } from '@tensorflow/tfjs-core'; // v4.x
import { TensorSpec, TensorMemoryInfo } from '../../types/tensor';
import { Logger } from '../utils/logger';

/**
 * Enhanced tensor memory manager with WebGL support and automatic cleanup
 * Maintains memory usage below 4GB while ensuring optimal performance
 */
export class TensorMemoryManager {
    private tensorRegistry: Map<string, TensorMemoryInfo>;
    private totalBytesUsed: number;
    private webglBytesUsed: number;
    private readonly maxBytesAllowed: number;
    private readonly cleanupThreshold: number;
    private readonly logger: Logger;
    private readonly tensorPriorities: Map<string, number>;
    private lastCleanupTime: number;

    constructor(
        maxBytesAllowed: number = 4 * 1024 * 1024 * 1024, // 4GB default
        cleanupThreshold: number = 0.85, // 85% threshold
        logger: Logger
    ) {
        this.tensorRegistry = new Map();
        this.maxBytesAllowed = maxBytesAllowed;
        this.cleanupThreshold = cleanupThreshold;
        this.totalBytesUsed = 0;
        this.webglBytesUsed = 0;
        this.logger = logger;
        this.tensorPriorities = new Map();
        this.lastCleanupTime = Date.now();

        // Monitor browser memory pressure
        if ('onmemorypressure' in window) {
            window.addEventListener('memorypressure', () => {
                this.handleMemoryPressure();
            });
        }
    }

    /**
     * Registers and tracks a tensor's memory usage with WebGL support
     * @param tensor - Tensor to track
     * @param spec - Tensor specification
     * @param priority - Priority level (higher = less likely to be cleaned up)
     * @returns Unique tensor ID
     */
    public trackTensor(tensor: tf.Tensor, spec: TensorSpec, priority: number = 1): string {
        const tensorId = `tensor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const byteSize = this.calculateTensorBytes(tensor);
        const webglMemory = this.getWebGLMemoryUsage(tensor);

        // Check memory constraints
        if (this.totalBytesUsed + byteSize > this.maxBytesAllowed) {
            this.logger.warn('Memory threshold exceeded, triggering cleanup');
            this.cleanupUnusedTensors();
        }

        // Register tensor
        this.tensorRegistry.set(tensorId, {
            byteSize,
            isDisposed: false,
            lastUsed: Date.now(),
            priority,
            webglMemory
        });

        this.tensorPriorities.set(tensorId, priority);
        this.totalBytesUsed += byteSize;
        this.webglBytesUsed += webglMemory;

        // Log memory allocation
        this.logger.logPerformance('tensor_allocation', {
            tensorId,
            byteSize,
            webglMemory,
            totalUsed: this.totalBytesUsed,
            webglUsed: this.webglBytesUsed
        });

        return tensorId;
    }

    /**
     * Disposes of a tensor and cleans up its resources
     * @param tensorId - ID of tensor to dispose
     * @returns Success status
     */
    public disposeTensor(tensorId: string): boolean {
        const tensorInfo = this.tensorRegistry.get(tensorId);
        if (!tensorInfo || tensorInfo.isDisposed) {
            return false;
        }

        try {
            // Release WebGL resources first
            if (tensorInfo.webglMemory > 0) {
                this.releaseWebGLResources(tensorId);
            }

            // Update memory counters
            this.totalBytesUsed -= tensorInfo.byteSize;
            this.webglBytesUsed -= tensorInfo.webglMemory;

            // Mark as disposed and remove from registry
            tensorInfo.isDisposed = true;
            this.tensorRegistry.delete(tensorId);
            this.tensorPriorities.delete(tensorId);

            this.logger.debug('Tensor disposed successfully', {
                tensorId,
                byteSize: tensorInfo.byteSize,
                webglMemory: tensorInfo.webglMemory
            });

            return true;
        } catch (error) {
            this.logger.error('Error disposing tensor', error);
            return false;
        }
    }

    /**
     * Gets comprehensive memory usage statistics
     * @returns Detailed memory statistics
     */
    public getMemoryInfo(): {
        totalBytesUsed: number;
        webglBytesUsed: number;
        numTensors: number;
        maxBytesAllowed: number;
        utilizationPercentage: number;
    } {
        const numTensors = this.tensorRegistry.size;
        const utilizationPercentage = (this.totalBytesUsed / this.maxBytesAllowed) * 100;

        return {
            totalBytesUsed: this.totalBytesUsed,
            webglBytesUsed: this.webglBytesUsed,
            numTensors,
            maxBytesAllowed: this.maxBytesAllowed,
            utilizationPercentage
        };
    }

    /**
     * Performs priority-based cleanup of unused tensors
     * @returns Number of tensors cleaned up
     */
    public cleanupUnusedTensors(): number {
        const now = Date.now();
        let cleanedCount = 0;

        // Sort tensors by priority and last used time
        const tensorEntries = Array.from(this.tensorRegistry.entries())
            .sort(([idA, infoA], [idB, infoB]) => {
                const priorityDiff = (this.tensorPriorities.get(idA) || 0) - (this.tensorPriorities.get(idB) || 0);
                if (priorityDiff !== 0) return priorityDiff;
                return infoA.lastUsed - infoB.lastUsed;
            });

        // Clean up tensors starting with lowest priority and oldest
        for (const [tensorId, info] of tensorEntries) {
            if (this.totalBytesUsed < this.maxBytesAllowed * this.cleanupThreshold) {
                break;
            }
            if (this.disposeTensor(tensorId)) {
                cleanedCount++;
            }
        }

        this.lastCleanupTime = now;
        this.logger.logPerformance('tensor_cleanup', {
            cleanedCount,
            totalBytesUsed: this.totalBytesUsed,
            webglBytesUsed: this.webglBytesUsed
        });

        return cleanedCount;
    }

    /**
     * Handles browser memory pressure events
     */
    private handleMemoryPressure(): void {
        this.logger.warn('Memory pressure detected, performing emergency cleanup');
        this.cleanupUnusedTensors();
    }

    /**
     * Calculates total bytes used by a tensor
     */
    private calculateTensorBytes(tensor: tf.Tensor): number {
        return tensor.size * Float32Array.BYTES_PER_ELEMENT;
    }

    /**
     * Gets WebGL memory usage for a tensor
     */
    private getWebGLMemoryUsage(tensor: tf.Tensor): number {
        // Estimate WebGL memory usage based on tensor size and data type
        return tensor.size * Float32Array.BYTES_PER_ELEMENT * 1.5; // 1.5x factor for WebGL overhead
    }

    /**
     * Releases WebGL resources for a tensor
     */
    private releaseWebGLResources(tensorId: string): void {
        const tensorInfo = this.tensorRegistry.get(tensorId);
        if (!tensorInfo) return;

        try {
            // Force WebGL context cleanup
            if (tf.env().getBool('WEBGL_VERSION')) {
                tf.engine().endScope();
                tf.engine().startScope();
            }
        } catch (error) {
            this.logger.error('Error releasing WebGL resources', error);
        }
    }
}