/**
 * @fileoverview Advanced browser Cache API wrapper for efficient temporary storage
 * @version 1.0.0
 * @license MIT
 */

import { Tensor } from '@tensorflow/tfjs-core'; // v4.x
import { CACHE_CONFIG } from '../../config/storage';
import { Logger } from '../utils/logger';

/**
 * Interface for cache metadata tracking
 */
interface CacheMetadata {
    key: string;
    size: number;
    timestamp: number;
    type: 'tensor' | 'frame' | 'buffer';
    lastAccessed: number;
    accessCount: number;
}

/**
 * Interface for cache operation options
 */
interface CacheOptions {
    maxAge?: number;
    priority?: 'high' | 'normal' | 'low';
    compression?: boolean;
}

/**
 * Advanced browser Cache API manager implementing sophisticated memory management
 */
export class CacheManager {
    private cache: Cache | null = null;
    private readonly logger: Logger;
    private readonly metadata: Map<string, CacheMetadata>;
    private totalSize: number = 0;
    private cleanupInterval: number;

    constructor(
        private readonly cacheName: string = CACHE_CONFIG.name,
        private readonly maxAge: number = CACHE_CONFIG.maxAge,
        private readonly maxItems: number = CACHE_CONFIG.maxItems
    ) {
        this.logger = new Logger({
            level: 'info',
            namespace: 'cache-manager',
            persistLogs: true
        });
        this.metadata = new Map();
        this.initializeCache();
    }

    /**
     * Initializes cache and sets up cleanup interval
     */
    private async initializeCache(): Promise<void> {
        try {
            this.cache = await caches.open(this.cacheName);
            this.cleanupInterval = window.setInterval(
                () => this.cleanup(),
                CACHE_CONFIG.memoryTracking.interval
            );
            await this.validateCacheQuota();
        } catch (error) {
            this.logger.error(error as Error, { component: 'CacheManager', operation: 'initialize' });
            throw error;
        }
    }

    /**
     * Stores data in Cache API with quota validation and cleanup
     */
    public async set(
        key: string,
        data: Blob | ArrayBuffer | Tensor,
        options: CacheOptions = {}
    ): Promise<void> {
        try {
            if (!this.cache) {
                throw new Error('Cache not initialized');
            }

            // Convert data to cacheable format
            const cacheData = await this.prepareCacheData(data);
            const size = this.calculateDataSize(cacheData);

            // Validate quota and cleanup if needed
            if (!await this.validateCacheQuota(size)) {
                await this.cleanup();
                if (!await this.validateCacheQuota(size)) {
                    throw new Error('Insufficient cache space');
                }
            }

            // Create cache response with metadata
            const metadata: CacheMetadata = {
                key,
                size,
                timestamp: Date.now(),
                type: data instanceof Tensor ? 'tensor' : 'buffer',
                lastAccessed: Date.now(),
                accessCount: 0
            };

            const headers = new Headers({
                'Cache-Control': `max-age=${options.maxAge || this.maxAge}`,
                'Content-Type': 'application/octet-stream',
                'X-Cache-Metadata': JSON.stringify(metadata)
            });

            const response = new Response(cacheData, { headers });
            await this.cache.put(key, response);

            // Update metadata tracking
            this.metadata.set(key, metadata);
            this.totalSize += size;

            this.logger.logPerformance('cache_write', size, 'memory');

        } catch (error) {
            this.logger.error(error as Error, { 
                component: 'CacheManager',
                operation: 'set',
                key,
                size: this.calculateDataSize(data)
            });
            throw error;
        }
    }

    /**
     * Retrieves and validates cached data
     */
    public async get(key: string): Promise<Response | null> {
        try {
            if (!this.cache) {
                throw new Error('Cache not initialized');
            }

            const response = await this.cache.match(key);
            if (!response) {
                return null;
            }

            const metadata = this.metadata.get(key);
            if (!metadata) {
                await this.cache.delete(key);
                return null;
            }

            // Validate age
            if (Date.now() - metadata.timestamp > this.maxAge) {
                await this.delete(key);
                return null;
            }

            // Update access statistics
            metadata.lastAccessed = Date.now();
            metadata.accessCount++;
            this.metadata.set(key, metadata);

            this.logger.logPerformance('cache_read', metadata.size, 'memory');

            return response;

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'CacheManager',
                operation: 'get',
                key
            });
            return null;
        }
    }

    /**
     * Removes item from cache with metadata cleanup
     */
    public async delete(key: string): Promise<boolean> {
        try {
            if (!this.cache) {
                throw new Error('Cache not initialized');
            }

            const metadata = this.metadata.get(key);
            if (metadata) {
                this.totalSize -= metadata.size;
                this.metadata.delete(key);
            }

            const result = await this.cache.delete(key);
            this.logger.logPerformance('cache_delete', metadata?.size || 0, 'memory');
            return result;

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'CacheManager',
                operation: 'delete',
                key
            });
            return false;
        }
    }

    /**
     * Clears all cached data with resource cleanup
     */
    public async clear(): Promise<void> {
        try {
            if (!this.cache) {
                throw new Error('Cache not initialized');
            }

            await caches.delete(this.cacheName);
            this.metadata.clear();
            this.totalSize = 0;
            this.cache = await caches.open(this.cacheName);

            this.logger.logPerformance('cache_clear', 0, 'memory');

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'CacheManager',
                operation: 'clear'
            });
            throw error;
        }
    }

    /**
     * Performs sophisticated cache cleanup with memory management
     */
    private async cleanup(): Promise<void> {
        try {
            const now = Date.now();
            const itemsToDelete: string[] = [];

            // Identify items for cleanup
            for (const [key, metadata] of this.metadata.entries()) {
                if (
                    now - metadata.timestamp > this.maxAge ||
                    now - metadata.lastAccessed > this.maxAge / 2
                ) {
                    itemsToDelete.push(key);
                }
            }

            // Enforce maximum items limit
            if (this.metadata.size > this.maxItems) {
                const sortedItems = Array.from(this.metadata.entries())
                    .sort((a, b) => a[1].lastAccessed - b[1].lastAccessed);
                
                const extraItems = sortedItems
                    .slice(0, this.metadata.size - this.maxItems)
                    .map(([key]) => key);
                
                itemsToDelete.push(...extraItems);
            }

            // Delete identified items
            await Promise.all(itemsToDelete.map(key => this.delete(key)));

            this.logger.logPerformance('cache_cleanup', itemsToDelete.length, 'memory');

        } catch (error) {
            this.logger.error(error as Error, {
                component: 'CacheManager',
                operation: 'cleanup'
            });
        }
    }

    /**
     * Validates cache quota availability
     */
    private async validateCacheQuota(requiredBytes: number = 0): Promise<boolean> {
        try {
            const estimate = await navigator.storage.estimate();
            const available = estimate.quota! - estimate.usage!;
            return available >= requiredBytes;
        } catch (error) {
            this.logger.error(error as Error, {
                component: 'CacheManager',
                operation: 'validateQuota'
            });
            return false;
        }
    }

    /**
     * Prepares data for caching
     */
    private async prepareCacheData(data: Blob | ArrayBuffer | Tensor): Promise<ArrayBuffer> {
        if (data instanceof Tensor) {
            const arrayBuffer = await data.data();
            return arrayBuffer.buffer;
        } else if (data instanceof Blob) {
            return await data.arrayBuffer();
        }
        return data;
    }

    /**
     * Calculates size of data for cache management
     */
    private calculateDataSize(data: any): number {
        if (data instanceof Tensor) {
            return data.size * Float32Array.BYTES_PER_ELEMENT;
        } else if (data instanceof Blob) {
            return data.size;
        } else if (data instanceof ArrayBuffer) {
            return data.byteLength;
        }
        return 0;
    }
}

export default CacheManager;