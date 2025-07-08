import React from 'react';
import { Typography, Box } from '@mui/material';

function TextOutput({ text, error = false }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom color={error ? 'error' : 'inherit'}>
        {error ? 'Error' : 'Generated Text'}
      </Typography>
      <Typography
        component="pre"
        sx={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          backgroundColor: error ? '#fee' : '#f5f5f5',
          p: 2,
          borderRadius: 1,
          maxHeight: '400px',
          overflowY: 'auto'
        }}
      >
        {text}
      </Typography>
    </Box>
  );
}

export default TextOutput;
