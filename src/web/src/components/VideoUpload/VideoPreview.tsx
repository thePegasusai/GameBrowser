/**
 * @fileoverview Memory-aware video preview component with WebGL acceleration
 * @version 1.0.0
 * @license MIT
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import { VideoFrame } from '../../types/video';
import { useVideo } from '../../hooks/useVideo';
import { MEMORY_CONSTRAINTS } from '../../constants/model';

// Memory usage thresholds in MB
const MEMORY_THRESHOLDS = {
  WARNING: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.7,
  CRITICAL: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 0.9
};

interface MemoryStats {
  totalUsed: number;
  gpuUsed: number;
  tensorCount: number;
}

interface VideoPreviewProps {
  file: File | null;
  onProcessingComplete: (frames: VideoFrame[], memoryStats: MemoryStats) => void;
  onMemoryPressure: (usage: number) => void;
}

// Styled components
const PreviewContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  maxWidth: '600px',
  aspectRatio: '16/9',
  position: 'relative',
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  backgroundColor: theme.palette.action.hover,
  transition: theme.transitions.create(['background-color', 'box-shadow']),
  '&:hover': {
    boxShadow: theme.shadows[4]
  }
}));

const VideoElement = styled('video')({
  width: '100%',
  height: '100%',
  objectFit: 'contain'
});

const ProcessingOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  color: theme.palette.common.white,
  zIndex: 1
}));

const MemoryIndicator = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: theme.spacing(2),
  right: theme.spacing(2),
  padding: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: 'rgba(0, 0, 0, 0.8)',
  color: theme.palette.common.white,
  fontSize: '0.75rem'
}));

/**
 * Video preview component with memory-aware processing and WebGL acceleration
 */
export const VideoPreview: React.FC<VideoPreviewProps> = ({
  file,
  onProcessingComplete,
  onMemoryPressure
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const {
    processVideo,
    state,
    frames,
    memoryUsage,
    reset
  } = useVideo({
    targetSize: [256, 256],
    frameStride: 1,
    memoryLimit: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE * 1024 * 1024
  });

  // Handle video loading
  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setObjectUrl(url);
      setError(null);
      
      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [file]);

  // Handle video load event
  const handleVideoLoad = useCallback(async () => {
    if (!videoRef.current || !file) return;

    try {
      // Check memory before processing
      if (memoryUsage.totalUsed > MEMORY_THRESHOLDS.CRITICAL) {
        throw new Error('Insufficient memory available for video processing');
      }

      // Start video processing
      await processVideo(file);

      // Notify parent of completion
      onProcessingComplete(frames, {
        totalUsed: memoryUsage.totalUsed,
        gpuUsed: memoryUsage.gpuUsed,
        tensorCount: memoryUsage.tensorCount
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Video processing failed');
      reset();
    }
  }, [file, processVideo, frames, memoryUsage, onProcessingComplete, reset]);

  // Monitor memory usage
  useEffect(() => {
    if (memoryUsage.totalUsed > MEMORY_THRESHOLDS.WARNING) {
      onMemoryPressure(memoryUsage.totalUsed);
    }
  }, [memoryUsage.totalUsed, onMemoryPressure]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      reset();
    };
  }, [reset]);

  return (
    <PreviewContainer>
      {file && (
        <VideoElement
          ref={videoRef}
          src={objectUrl || undefined}
          onLoadedMetadata={handleVideoLoad}
          controls={state !== 'processing'}
          muted
          playsInline
        />
      )}

      {state === 'processing' && (
        <ProcessingOverlay>
          <CircularProgress color="inherit" size={48} />
          <Typography variant="body2" sx={{ mt: 2 }}>
            Processing video...
          </Typography>
        </ProcessingOverlay>
      )}

      {error && (
        <Alert 
          severity="error" 
          sx={{ 
            position: 'absolute', 
            bottom: 0, 
            left: 0, 
            right: 0 
          }}
        >
          {error}
        </Alert>
      )}

      <MemoryIndicator>
        Memory: {Math.round(memoryUsage.totalUsed / 1024 / 1024)}MB
        {memoryUsage.gpuUsed > 0 && ` (GPU: ${Math.round(memoryUsage.gpuUsed / 1024 / 1024)}MB)`}
      </MemoryIndicator>
    </PreviewContainer>
  );
};

export default VideoPreview;