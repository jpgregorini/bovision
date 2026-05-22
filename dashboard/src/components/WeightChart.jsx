import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function WeightChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">Historico de Pesagens</h2>
        <p className="text-gray-400">Sem dados ainda.</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">Historico de Pesagens</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} label={{ value: 'kg', position: 'insideLeft', offset: -5 }} />
          <Tooltip formatter={(value) => [`${value} kg`, 'Peso Medio']} labelFormatter={(label) => `Data: ${label}`} />
          <Line type="monotone" dataKey="average_weight_kg" stroke="#16a34a" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 6 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
