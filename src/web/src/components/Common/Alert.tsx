/**
 * Alert component for browser-based video game diffusion model interface
 * Implements Material Design principles with memory-aware auto-dismiss
 * @version 1.0.0
 */

import React, { useEffect, useCallback } from 'react';
import styled from 'styled-components'; // v5.3.0
import { theme } from '../../styles/theme';
import { Logger } from '../../lib/utils/logger';

// Initialize logger for alert events
const logger = new Logger({
  level: 'info',
  namespace: 'alert-component',
  persistLogs: false,
  metricsRetentionMs: 3600000
});

/**
 * Props interface for Alert component with enhanced accessibility
 */
interface AlertProps {
  children: React.ReactNode;
  severity: 'error' | 'warning' | 'info' | 'success';
  dismissible?: boolean;
  autoHideDuration?: number;
  onClose?: () => void;
  role?: string;
  ariaLive?: 'polite' | 'assertive';
}

/**
 * Styled container for alert with enhanced accessibility and theme-based styling
 */
const AlertContainer = styled.div<{ severity: AlertProps['severity'] }>`
  display: flex;
  align-items: center;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 4px;
  background-color: ${props => getBackgroundColor(props.severity)};
  color: ${props => getTextColor(props.severity)};
  font-family: ${theme.typography.fontFamily};
  font-size: 14px;
  line-height: 1.5;
  transition: opacity 0.3s ease-in-out;
  position: relative;
  max-width: 100%;
  box-sizing: border-box;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  &:focus-within {
    outline: 2px solid ${props => getTextColor(props.severity)};
    outline-offset: 2px;
  }
`;

/**
 * Styled close button with enhanced accessibility
 */
const CloseButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  margin-left: auto;
  color: inherit;
  opacity: 0.7;
  transition: opacity 0.2s ease-in-out;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    opacity: 1;
  }

  &:focus {
    outline: 2px solid currentColor;
    outline-offset: 2px;
  }
`;

/**
 * Gets theme-appropriate background color based on severity
 */
const getBackgroundColor = (severity: AlertProps['severity']): string => {
  const alpha = 0.1; // Ensure WCAG contrast compliance
  switch (severity) {
    case 'error':
      return `${theme.palette.error.main}${Math.round(alpha * 255).toString(16)}`;
    case 'warning':
      return `${theme.palette.warning.main}${Math.round(alpha * 255).toString(16)}`;
    case 'success':
      return `${theme.palette.success.main}${Math.round(alpha * 255).toString(16)}`;
    case 'info':
    default:
      return `${theme.palette.primary.main}${Math.round(alpha * 255).toString(16)}`;
  }
};

/**
 * Gets accessible text color based on severity
 */
const getTextColor = (severity: AlertProps['severity']): string => {
  switch (severity) {
    case 'error':
      return theme.palette.error.main;
    case 'warning':
      return theme.palette.warning.main;
    case 'success':
      return theme.palette.success.main;
    case 'info':
    default:
      return theme.palette.primary.main;
  }
};

/**
 * Memory-efficient alert component with accessibility support
 */
export const Alert: React.FC<AlertProps> = React.memo(({
  children,
  severity = 'info',
  dismissible = true,
  autoHideDuration,
  onClose,
  role = 'alert',
  ariaLive = severity === 'error' ? 'assertive' : 'polite'
}) => {
  // Handle alert dismissal with memory cleanup
  const handleClose = useCallback(() => {
    logger.log('Alert dismissed', 'info', { severity });
    if (onClose) {
      onClose();
    }
  }, [onClose, severity]);

  // Auto-dismiss with memory management
  useEffect(() => {
    let timeoutId: number;

    if (autoHideDuration && autoHideDuration > 0) {
      timeoutId = window.setTimeout(() => {
        handleClose();
      }, autoHideDuration);

      // Log alert display with memory context
      logger.log('Alert displayed', 'info', {
        severity,
        autoHideDuration,
        memoryUsage: performance.memory?.usedJSHeapSize
      });
    }

    // Cleanup timeout and log on unmount
    return () => {
      if (timeoutId) {
        window.clearTimeout(timeoutId);
        logger.log('Alert cleanup', 'debug', { severity });
      }
    };
  }, [autoHideDuration, handleClose, severity]);

  return (
    <AlertContainer
      severity={severity}
      role={role}
      aria-live={ariaLive}
      aria-atomic="true"
    >
      {children}
      {dismissible && (
        <CloseButton
          onClick={handleClose}
          aria-label="Close alert"
          type="button"
        >
          ×
        </CloseButton>
      )}
    </AlertContainer>
  );
});

Alert.displayName = 'Alert';

export default Alert;