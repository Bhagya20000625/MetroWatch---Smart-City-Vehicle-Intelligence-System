import { useState, useEffect } from 'react';
import { api } from './services/api';
import StatsCard from './components/StatsCard';
import VehicleTypeChart from './components/VehicleTypeChart';
import ProvinceChart from './components/ProvinceChart';
import RecentVehiclesTable from './components/RecentVehiclesTable';
import ImageUpload from './components/ImageUpload';

function App() {
  const [analytics, setAnalytics] = useState(null);
  const [recentVehicles, setRecentVehicles] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [analyticsRes, vehiclesRes] = await Promise.all([
        api.getAnalytics(),
        api.getRecentVehicles(10)
      ]);
      setAnalytics(analyticsRes.data);
      setRecentVehicles(vehiclesRes.data.vehicles || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const handleReset = async () => {
    if (window.confirm('Reset all analytics data?')) {
      try {
        await api.resetAnalytics();
        fetchData();
      } catch (error) {
        alert('Reset failed: ' + error.message);
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl text-red-600">Failed to load data. Make sure the API server is running.</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-900">
              🚗 MetroWatch Dashboard
            </h1>
            <button
              onClick={handleReset}
              className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
            >
              Reset Analytics
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 px-4">
        {/* Image Upload */}
        <div className="mb-6">
          <ImageUpload onDetectionComplete={fetchData} />
        </div>

        {/* Analytics Section Header */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">
                📊 All-Time Analytics (Cumulative Data)
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>These statistics show the <strong>total accumulated data</strong> from all videos and images you've uploaded since the last reset. Each upload adds to these totals.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <StatsCard
            title="Total Vehicles"
            value={analytics.total_vehicles}
            icon="🚗"
            color="border-blue-500"
            subtitle="All time"
          />
          <StatsCard
            title="Total Entries"
            value={analytics.total_entries}
            icon="📥"
            color="border-green-500"
            subtitle="All time"
          />
          <StatsCard
            title="Total Exits"
            value={analytics.total_exits}
            icon="📤"
            color="border-orange-500"
            subtitle="All time"
          />
          <StatsCard
            title="Traffic Flow"
            value={analytics.total_traffic_flow}
            icon="🚦"
            color="border-purple-500"
            subtitle="All time"
          />
        </div>

        {/* Charts Section */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">📈 Distribution Analysis</h2>
        </div>
        
        {/* Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <VehicleTypeChart data={analytics.by_type} />
          <ProvinceChart data={analytics.by_province} />
        </div>

        {/* Recent Vehicles Section */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">🕐 Recent Detections</h2>
        </div>
        
        {/* Recent Vehicles Table */}
        <RecentVehiclesTable vehicles={recentVehicles} />
      </main>
    </div>
  );
}


export default App;