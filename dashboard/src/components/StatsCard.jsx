export default function StatsCard({ title, value, icon, color, subtitle }) {
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 border-l-4 ${color}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-500 text-sm font-medium">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
          )}
        </div>
        <div className="text-4xl">{icon}</div>
      </div>
    </div>
  );
}