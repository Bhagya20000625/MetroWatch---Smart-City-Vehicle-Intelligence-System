import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const api = {
  // Health check
  healthCheck: () => axios.get(`${API_BASE_URL}/health`),
  
  // Upload image for detection
  detectVehicles: (imageFile) => {
    const formData = new FormData();
    formData.append('file', imageFile);
    return axios.post(`${API_BASE_URL}/detect`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  
  // Get analytics summary
  getAnalytics: () => axios.get(`${API_BASE_URL}/analytics/summary`),
  
  // Get recent vehicles
  getRecentVehicles: (limit = 10) => 
    axios.get(`${API_BASE_URL}/analytics/vehicles?limit=${limit}`),
  
  // Reset analytics
  resetAnalytics: () => axios.post(`${API_BASE_URL}/analytics/reset`)
};