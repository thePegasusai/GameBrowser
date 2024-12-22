/**
 * @fileoverview Enhanced React hook for browser storage management with memory tracking
 * @version 1.0.0
 * @license MIT
 */

import { useState, useEffect, useCallback } from 'react'; // v18.0.0
import { Tensor } from '@tensorflow/tfjs-core'; // v4.x
import { 
  INDEXED_DB_CONFIG, 
  CACHE_CONFIG, 
  STORAGE_QUOTAS, 
  StorageValidation 
} from '../../config/storage';

/**
 * Interface for storage statistics with memory tracking
 */
interface StorageStats {
  indexedDB: {
    used: number;
    available: number;
    modelCount: number;
  };
  cache: {
    used: number;
    itemCount: number;
    oldestItem: Date;
  };
  memory: {
    heapUsed: number;
    gpuUsed: number | null;
  };
}

/**
 * Interface for storage quota information
 */
interface StorageQuota {
  total: number;
  used: number;
  available: number;
  isLow: boolean;
}

/**
 * Enhanced storage hook result interface
 */
interface StorageHookResult {
  isInitialized: boolean;
  error: Error | null;
  stats: StorageStats;
  saveModelWeights: (id: string, weights: Tensor[]) => Promise<void>;
  loadModelWeights: (id: string) => Promise<Tensor[]>;
  cacheVideoFrame: (key: string, frame: Blob) => Promise<void>;
  getCachedFrame: (key: string) => Promise<Blob | null>;
  clearCache: () => Promise<void>;
  optimizeStorage: () => Promise<void>;
  checkQuota: () => Promise<StorageQuota>;
}

/**
 * Enhanced custom hook for managing browser storage with memory tracking
 * @returns {StorageHookResult} Storage operations and state
 */
export function useStorage(): StorageHookResult {
  // State management
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [stats, setStats] = useState<StorageStats>({
    indexedDB: { used: 0, available: 0, modelCount: 0 },
    cache: { used: 0, itemCount: 0, oldestItem: new Date() },
    memory: { heapUsed: 0, gpuUsed: null }
  });

  /**
   * Initializes storage systems with validation
   */
  const initializeStorage = useCallback(async () => {
    try {
      // Validate storage availability
      const availability = await StorageValidation.checkStorageAvailability();
      if (!availability.indexedDB || !availability.cacheAPI) {
        throw new Error('Required storage mechanisms unavailable');
      }

      // Validate storage quota
      const hasQuota = await StorageValidation.validateQuota();
      if (!hasQuota) {
        throw new Error('Insufficient storage quota');
      }

      setIsInitialized(true);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Storage initialization failed'));
    }
  }, []);

  /**
   * Updates storage statistics
   */
  const updateStats = useCallback(async () => {
    try {
      const estimate = await navigator.storage.estimate();
      const idbStats = await getIndexedDBStats();
      const cacheStats = await getCacheStats();

      setStats({
        indexedDB: idbStats,
        cache: cacheStats,
        memory: {
          heapUsed: performance.memory?.usedJSHeapSize || 0,
          gpuUsed: await getGPUMemoryUsage()
        }
      });
    } catch (err) {
      console.warn('Failed to update storage stats:', err);
    }
  }, []);

  /**
   * Saves model weights to IndexedDB with validation
   */
  const saveModelWeights = useCallback(async (id: string, weights: Tensor[]) => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      // Validate weights size
      const totalSize = weights.reduce((size, tensor) => 
        size + (tensor.size * Float32Array.BYTES_PER_ELEMENT), 0);
      
      if (totalSize > STORAGE_QUOTAS.MAX_MODEL_SIZE) {
        throw new Error('Model weights exceed size limit');
      }

      const db = await openIndexedDB();
      const transaction = db.transaction([INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS], 'readwrite');
      const store = transaction.objectStore(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS);

      await store.put({
        id,
        weights: weights.map(t => t.arraySync()),
        timestamp: Date.now()
      });

      await updateStats();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to save model weights'));
      throw err;
    }
  }, [isInitialized, updateStats]);

  /**
   * Loads model weights from IndexedDB
   */
  const loadModelWeights = useCallback(async (id: string): Promise<Tensor[]> => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      const db = await openIndexedDB();
      const transaction = db.transaction([INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS], 'readonly');
      const store = transaction.objectStore(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS);
      
      const result = await store.get(id);
      if (!result) throw new Error('Model weights not found');

      return result.weights.map((data: any[]) => Tensor.make(data));
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load model weights'));
      throw err;
    }
  }, [isInitialized]);

  /**
   * Caches video frame with cleanup checks
   */
  const cacheVideoFrame = useCallback(async (key: string, frame: Blob) => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      const cache = await caches.open(CACHE_CONFIG.name);
      const response = new Response(frame);
      await cache.put(key, response);

      const stats = await getCacheStats();
      if (stats.used > CACHE_CONFIG.maxSize) {
        await optimizeCache();
      }

      await updateStats();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to cache video frame'));
      throw err;
    }
  }, [isInitialized, updateStats]);

  /**
   * Retrieves cached video frame
   */
  const getCachedFrame = useCallback(async (key: string): Promise<Blob | null> => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      const cache = await caches.open(CACHE_CONFIG.name);
      const response = await cache.match(key);
      
      return response ? await response.blob() : null;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to retrieve cached frame'));
      throw err;
    }
  }, [isInitialized]);

  /**
   * Clears cache with validation
   */
  const clearCache = useCallback(async () => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      await caches.delete(CACHE_CONFIG.name);
      await updateStats();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to clear cache'));
      throw err;
    }
  }, [isInitialized, updateStats]);

  /**
   * Optimizes storage usage
   */
  const optimizeStorage = useCallback(async () => {
    try {
      if (!isInitialized) throw new Error('Storage not initialized');

      await Promise.all([
        optimizeCache(),
        cleanupOldModels()
      ]);

      await updateStats();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to optimize storage'));
      throw err;
    }
  }, [isInitialized, updateStats]);

  /**
   * Checks storage quota status
   */
  const checkQuota = useCallback(async (): Promise<StorageQuota> => {
    try {
      const estimate = await navigator.storage.estimate();
      const used = estimate.usage || 0;
      const total = estimate.quota || 0;
      const available = total - used;
      
      return {
        total,
        used,
        available,
        isLow: available < STORAGE_QUOTAS.MIN_AVAILABLE
      };
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to check quota'));
      throw err;
    }
  }, []);

  // Initialize storage on mount
  useEffect(() => {
    initializeStorage();

    // Set up periodic stats updates
    const statsInterval = setInterval(updateStats, CACHE_CONFIG.memoryTracking.interval);

    // Cleanup on unmount
    return () => {
      clearInterval(statsInterval);
    };
  }, [initializeStorage, updateStats]);

  return {
    isInitialized,
    error,
    stats,
    saveModelWeights,
    loadModelWeights,
    cacheVideoFrame,
    getCachedFrame,
    clearCache,
    optimizeStorage,
    checkQuota
  };
}

// Utility functions
async function openIndexedDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(INDEXED_DB_CONFIG.name, INDEXED_DB_CONFIG.version);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

async function getIndexedDBStats(): Promise<StorageStats['indexedDB']> {
  const db = await openIndexedDB();
  const transaction = db.transaction([INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS], 'readonly');
  const store = transaction.objectStore(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS);
  
  return new Promise((resolve) => {
    const request = store.count();
    request.onsuccess = () => resolve({
      used: 0, // Will be updated by storage estimate
      available: 0,
      modelCount: request.result
    });
  });
}

async function getCacheStats(): Promise<StorageStats['cache']> {
  const cache = await caches.open(CACHE_CONFIG.name);
  const keys = await cache.keys();
  let oldestDate = new Date();
  let totalSize = 0;

  for (const request of keys) {
    const response = await cache.match(request);
    if (response) {
      const blob = await response.blob();
      totalSize += blob.size;
      const date = new Date(response.headers.get('date') || Date.now());
      if (date < oldestDate) oldestDate = date;
    }
  }

  return {
    used: totalSize,
    itemCount: keys.length,
    oldestItem: oldestDate
  };
}

async function optimizeCache(): Promise<void> {
  const cache = await caches.open(CACHE_CONFIG.name);
  const keys = await cache.keys();
  const now = Date.now();

  for (const request of keys) {
    const response = await cache.match(request);
    if (response) {
      const date = new Date(response.headers.get('date') || now);
      if (now - date.getTime() > CACHE_CONFIG.maxAge) {
        await cache.delete(request);
      }
    }
  }
}

async function cleanupOldModels(): Promise<void> {
  const db = await openIndexedDB();
  const transaction = db.transaction([INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS], 'readwrite');
  const store = transaction.objectStore(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS);
  
  const request = store.openCursor();
  const now = Date.now();

  return new Promise((resolve) => {
    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor) {
        const timestamp = cursor.value.timestamp;
        if (now - timestamp > STORAGE_QUOTAS.MONITORING.CHECK_INTERVAL) {
          cursor.delete();
        }
        cursor.continue();
      } else {
        resolve();
      }
    };
  });
}

async function getGPUMemoryUsage(): Promise<number | null> {
  try {
    // @ts-ignore: Chrome-specific API
    if (performance.memory?.gpu) {
      // @ts-ignore
      return performance.memory.gpu.usedGPUMemory;
    }
    return null;
  } catch {
    return null;
  }
}