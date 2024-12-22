/**
 * @fileoverview Main Generation component for video game footage generation with memory optimization
 * @version 1.0.0
 * @license MIT
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { ActionControls, ActionControlsProps } from './ActionControls';
import { GenerationControls, GenerationControlsProps, MemoryStats } from './Controls';
import { Preview, PreviewProps } from './Preview';
import { useGeneration } from '../../hooks/useGeneration';

// Styled components for layout
const GenerationContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  background-color: var(--background-primary);
  border-radius: 12px;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
`;

const PreviewSection = styled.div`
  width: 100%;
  aspect-ratio: 16/9;
  position: relative;
`;

const ControlsSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const MemoryIndicator = styled.div<{ warning: boolean }>`
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 8px;
  border-radius: 4px;
  background-color: ${props => props.warning ? 'var(--error-light)' : 'var(--background-overlay)'};
  color: ${props => props.warning ? 'var(--error-dark)' : 'var(--text-primary)'};
  font-family: 'Roboto Mono', monospace;
  font-size: 12px;
  z-index: 100;
`;

// Component props interface
interface GenerationProps {
  initialFrame: tf.Tensor4D | null;
  onError: (error: Error) => void;
  onMemoryWarning: (stats: MemoryStats) => void;
}

// Generation state interface
interface GenerationState {
  currentActions: Record<string, number>;
  generationParams: {
    actionStrength: number;
    noiseLevel: number;
    timesteps: number;
    memoryLimit: number;
  };
  memoryStats: MemoryStats;
}

/**
 * Main Generation component that orchestrates video game footage generation
 * with memory management and performance optimization
 */
export const Generation: React.FC<GenerationProps> = React.memo(({
  initialFrame,
  onError,
  onMemoryWarning
}) => {
  // State management
  const [state, setState] = useState<GenerationState>({
    currentActions: {},
    generationParams: {
      actionStrength: 0.5,
      noiseLevel: 0.1,
      timesteps: 20,
      memoryLimit: 3072 // 3GB default
    },
    memoryStats: {
      used: 0,
      total: 4096,
      webgl: 0
    }
  });

  // Generation hook with memory monitoring
  const {
    generateFrame,
    isGenerating,
    error: generationError,
    memoryStats
  } = useGeneration({
    batchSize: 1,
    temperature: state.generationParams.noiseLevel,
    useCache: true,
    memoryOptimize: true
  });

  // Performance monitoring
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(Date.now());

  /**
   * Handles action changes with memory optimization
   */
  const handleActionChange = useCallback((actionType: string, value: number) => {
    setState(prev => ({
      ...prev,
      currentActions: {
        ...prev.currentActions,
        [actionType]: value
      }
    }));
  }, []);

  /**
   * Handles generation parameter updates with validation
   */
  const handleGenerationParamsChange = useCallback((params: Partial<GenerationState['generationParams']>) => {
    setState(prev => ({
      ...prev,
      generationParams: {
        ...prev.generationParams,
        ...params
      }
    }));
  }, []);

  /**
   * Monitors memory usage and triggers warnings
   */
  useEffect(() => {
    if (memoryStats) {
      setState(prev => ({ ...prev, memoryStats }));
      
      const memoryUsagePercent = (memoryStats.used / memoryStats.total) * 100;
      if (memoryUsagePercent > 85) {
        onMemoryWarning(memoryStats);
      }
    }
  }, [memoryStats, onMemoryWarning]);

  /**
   * Handles errors from generation process
   */
  useEffect(() => {
    if (generationError) {
      onError(generationError);
    }
  }, [generationError, onError]);

  /**
   * Handles frame generation with performance monitoring
   */
  const handleGenerate = useCallback(async () => {
    if (!initialFrame) return;

    const currentTime = Date.now();
    const frameTime = currentTime - lastFrameTimeRef.current;
    
    try {
      const generatedFrame = await generateFrame(
        initialFrame,
        state.currentActions,
        state.generationParams.timesteps
      );

      frameCountRef.current++;
      lastFrameTimeRef.current = currentTime;

      return generatedFrame;
    } catch (error) {
      onError(error as Error);
    }
  }, [initialFrame, generateFrame, state.currentActions, state.generationParams.timesteps, onError]);

  return (
    <GenerationContainer>
      <MemoryIndicator warning={memoryStats.used / memoryStats.total > 0.85}>
        Memory: {Math.round(memoryStats.used)}MB / {memoryStats.total}MB
      </MemoryIndicator>

      <PreviewSection>
        <Preview
          width={1280}
          height={720}
          onError={onError}
          quality={{
            targetPSNR: 30,
            adaptiveScaling: true,
            maxFrameLatency: 50
          }}
          performance={{
            enableFrameSkipping: true,
            targetFPS: 30,
            memoryLimit: state.generationParams.memoryLimit
          }}
        />
      </PreviewSection>

      <ControlsSection>
        <ActionControls
          onActionChange={handleActionChange}
          disabled={!initialFrame || isGenerating}
          currentActions={state.currentActions}
          memoryConfig={{
            maxMemoryUsage: state.generationParams.memoryLimit,
            warningThreshold: 0.85
          }}
        />
        <GenerationControls
          onGenerate={handleGenerate}
          onStop={() => {}}
          disabled={!initialFrame}
          onError={onError}
          onMemoryWarning={onMemoryWarning}
        />
      </ControlsSection>
    </GenerationContainer>
  );
});

Generation.displayName = 'Generation';

export type { GenerationProps, GenerationState, MemoryStats };
export default Generation;