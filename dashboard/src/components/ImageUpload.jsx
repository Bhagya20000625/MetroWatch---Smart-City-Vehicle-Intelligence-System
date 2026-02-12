import { useState } from 'react';
import api from '../services/api';

const ImageUpload = ({ onDetectionComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'video'

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Clear previous state
      setResult(null);
      setPreview(null);
      setFileType(null);
      
      setSelectedFile(file);
      
      // Determine file type
      if (file.type.startsWith('image/')) {
        setFileType('image');
        setPreview(URL.createObjectURL(file));
        console.log('Selected image file:', file.name);
      } else if (file.type.startsWith('video/')) {
        setFileType('video');
        setPreview(URL.createObjectURL(file));
        console.log('Selected video file:', file.name);
      } else {
        console.log('Unknown file type:', file.type);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setResult(null);

    try {
      let response;
      console.log('=== UPLOAD DEBUG ===');
      console.log('File name:', selectedFile.name);
      console.log('File type:', selectedFile.type);
      console.log('Detected fileType state:', fileType);
      
      if (fileType === 'image') {
        console.log('Calling detectVehicles (IMAGE)');
        response = await api.detectVehicles(selectedFile);
      } else if (fileType === 'video') {
        console.log('Calling detectVehiclesVideo (VIDEO)');
        response = await api.detectVehiclesVideo(selectedFile);
      } else {
        throw new Error('Unknown file type');
      }
      
      console.log('Response received:', response.data);
      console.log('Response has vehicle_count?', response.data.vehicle_count !== undefined);
      console.log('Response has total_vehicles?', response.data.total_vehicles !== undefined);
      
      setResult(response.data);
      onDetectionComplete();
    } catch (error) {
      console.error('Upload error:', error);
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Upload failed'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">📤 Upload Media</h2>
        {(selectedFile || result) && (
          <button
            onClick={() => {
              setSelectedFile(null);
              setPreview(null);
              setFileType(null);
              setResult(null);
            }}
            className="text-sm px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded"
          >
            Clear
          </button>
        )}
      </div>
      
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
        <input
          type="file"
          accept="image/*,video/*"
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer text-blue-600 hover:text-blue-800"
        >
          Click to select image or video
        </label>
        
        {preview && (
          <div className="mt-4">
            {fileType === 'image' ? (
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded"
              />
            ) : (
              <video
                src={preview}
                controls
                className="max-h-64 mx-auto rounded"
              />
            )}
          </div>
        )}
        
        {selectedFile && (
          <button
            onClick={handleUpload}
            disabled={isProcessing}
            className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400"
          >
            {isProcessing ? 'Processing...' : `Detect Vehicles in ${fileType === 'image' ? 'Image' : 'Video'}`}
          </button>
        )}
      </div>

      {result && (
        <div className={`mt-4 p-4 rounded ${result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
          {result.success ? (
            <div>
              <p className="text-green-800 font-semibold">✓ {result.message || 'Detection complete!'}</p>
              
              {/* Image result - has vehicle_count */}
              {result.vehicle_count !== undefined && (
                <p className="text-green-700">Found {result.vehicle_count} vehicle(s)</p>
              )}
              
              {/* Video result - has total_vehicles, processed_frames, etc */}
              {result.total_vehicles !== undefined && (
                <div className="text-green-700 mt-2">
                  <p>📊 <strong>Total Vehicles Tracked:</strong> {result.total_vehicles}</p>
                  <p>🎬 <strong>Frames Processed:</strong> {result.processed_frames}/{result.total_frames}</p>
                  <p>📥 <strong>Total Entries:</strong> {result.total_entries}</p>
                  <p>📤 <strong>Total Exits:</strong> {result.total_exits}</p>
                  <p>🎥 <strong>FPS:</strong> {result.fps}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-red-800">✗ {result.message}</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;