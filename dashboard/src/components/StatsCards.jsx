function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-gray-900">{value ?? '—'}</p>
      {sub && <p className="mt-1 text-xs text-gray-400">{sub}</p>}
    </div>
  );
}

export default function StatsCards({ summary }) {
  if (!summary) return null;
  const lastAt = summary.last_weighing_at
    ? new Date(summary.last_weighing_at).toLocaleString('pt-BR')
    : 'Nenhuma';
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <StatCard label="Total de Pesagens" value={summary.total_weighings} />
      <StatCard label="Peso Medio" value={summary.average_weight_kg ? `${summary.average_weight_kg} kg` : null} />
      <StatCard label="Ultima Pesagem" value={lastAt} />
      <StatCard label="Alertas Ativos" value={summary.active_alerts} sub={summary.active_alerts > 0 ? 'Animais com perda de peso' : null} />
    </div>
  );
}
