import React from 'react'; // ^18.0.0
import styled from 'styled-components'; // ^5.3.0
import { useMediaQuery } from '@mui/material'; // ^5.3.0

// Internal imports
import { lightTheme as theme } from '../../styles/theme';

/**
 * Props interface for Footer component
 */
interface FooterProps {
  version: string;
}

/**
 * Styled footer container with responsive layout and theme support
 */
const FooterContainer = styled.footer<{ isMobile: boolean }>`
  width: 100%;
  padding: ${({ theme }) => theme.spacing(2)};
  background-color: ${({ theme }) => theme.palette.background.paper};
  color: ${({ theme }) => theme.palette.text.primary};
  display: flex;
  flex-direction: ${({ isMobile }) => isMobile ? 'column' : 'row'};
  justify-content: space-between;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(2)};
  transition: background-color 0.3s ease, color 0.3s ease;
  border-top: 1px solid ${({ theme }) => theme.palette.divider};
  min-height: ${({ theme }) => theme.spacing(6)};
`;

/**
 * Styled footer content section with flexible layout
 */
const FooterContent = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing(2)};
  align-items: center;
  justify-content: center;
`;

/**
 * Styled footer text with theme-aware typography
 */
const FooterText = styled.p`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.body2.fontSize};
  color: ${({ theme }) => theme.palette.text.secondary};
  line-height: 1.5;
  font-family: ${({ theme }) => theme.typography.fontFamily};
`;

/**
 * Styled footer link with accessibility and interaction states
 */
const FooterLink = styled.a`
  color: ${({ theme }) => theme.palette.primary.main};
  text-decoration: none;
  transition: color 0.2s ease, transform 0.2s ease;
  padding: ${({ theme }) => theme.spacing(1)};
  border-radius: 4px;
  font-family: ${({ theme }) => theme.typography.fontFamily};

  &:hover {
    color: ${({ theme }) => theme.palette.primary.dark};
    transform: translateY(-1px);
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.palette.primary.main};
    outline-offset: 2px;
  }

  &:active {
    transform: translateY(0);
  }
`;

/**
 * Footer component displaying application information and links
 * Implements Material Design principles and accessibility features
 * @param {FooterProps} props - Component props
 * @returns {JSX.Element} Themed and accessible footer
 */
const Footer: React.FC<FooterProps> = React.memo(({ version }) => {
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const currentYear = new Date().getFullYear();

  return (
    <FooterContainer 
      isMobile={isMobile}
      role="contentinfo"
      aria-label="Application footer"
    >
      <FooterContent>
        <FooterText>
          © {currentYear} Browser-based Video Game Diffusion Model
        </FooterText>
        <FooterText aria-label="Application version">
          Version: {version}
        </FooterText>
      </FooterContent>

      <FooterContent>
        <FooterLink
          href="https://github.com/yourusername/bvgdm"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="View source code on GitHub"
        >
          GitHub
        </FooterLink>
        <FooterLink
          href="/docs"
          aria-label="View documentation"
        >
          Documentation
        </FooterLink>
        <FooterLink
          href="/privacy"
          aria-label="View privacy policy"
        >
          Privacy
        </FooterLink>
      </FooterContent>
    </FooterContainer>
  );
});

// Display name for debugging
Footer.displayName = 'Footer';

export default Footer;