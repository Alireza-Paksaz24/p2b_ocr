import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone'; // we can add react-dropzone, but let's keep it simple without extra dep
// We'll implement a basic drag-and-drop using native events.

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  accept: string;
  className?: string;
}

export function FileUpload({ onFilesSelected, accept, className = '' }: FileUploadProps) {
  const [dragActive, setDragActive] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        onFilesSelected(Array.from(e.dataTransfer.files));
      }
    },
    [onFilesSelected]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files));
      // reset input value so same file can be re-selected
      e.target.value = '';
    }
  };

  return (
    <div
      className={`relative border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
        dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      } ${className}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={accept}
        multiple
        onChange={handleChange}
      />
      <svg
        className="w-8 h-8 text-gray-400 mb-2"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 6v6m0 0v6m0-6h6m-6 0H6"
        />
      </svg>
      <p className="text-sm text-gray-500">
        Drag & drop files here, or click to select
      </p>
      <p className="text-xs text-gray-400 mt-1">
        Accepted: images, PDF, ZIP
      </p>
    </div>
  );
}