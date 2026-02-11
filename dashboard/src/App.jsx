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
      setRecentVehicles(vehiclesRes.data);
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
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <StatsCard
            title="Total Vehicles"
            value={analytics.total_vehicles}
            icon="🚗"
            color="border-blue-500"
          />
          <StatsCard
            title="Entries"
            value={analytics.total_entries}
            icon="📥"
            color="border-green-500"
          />
          <StatsCard
            title="Exits"
            value={analytics.total_exits}
            icon="📤"
            color="border-orange-500"
          />
          <StatsCard
            title="Current Count"
            value={analytics.current_count}
            icon="🔢"
            color="border-purple-500"
          />
        </div>

        {/* Image Upload */}
        <div className="mb-6">
          <ImageUpload onDetectionComplete={fetchData} />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <VehicleTypeChart data={analytics.by_type} />
          <ProvinceChart data={analytics.by_province} />
        </div>

        {/* Recent Vehicles Table */}
        <RecentVehiclesTable vehicles={recentVehicles} />
      </main>
    </div>
  );
}

export default App;