import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, CircularProgress } from '@mui/material';

function FileUpload({ onUpload, isLoading }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/*': ['.txt', '.doc', '.docx', '.pdf'],
    },
    multiple: false
  });

  return (
    <Box
      {...getRootProps()}
      sx={{
        border: '2px dashed #ccc',
        borderRadius: 2,
        p: 3,
        textAlign: 'center',
        cursor: 'pointer',
        backgroundColor: isDragActive ? '#f0f0f0' : 'transparent',
        '&:hover': {
          backgroundColor: '#f0f0f0'
        }
      }}
    >
      <input {...getInputProps()} />
      {isLoading ? (
        <CircularProgress />
      ) : (
        <>
          <Typography variant="h6" gutterBottom>
            Drag & drop a file here
          </Typography>
          <Typography color="text.secondary">
            or click to select a file
          </Typography>
        </>
      )}
    </Box>
  );
}

export default FileUpload;
