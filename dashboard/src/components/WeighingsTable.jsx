export default function WeighingsTable({ weighings }) {
  if (!weighings || weighings.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">Pesagens Recentes</h2>
        <p className="text-gray-400">Nenhuma pesagem registrada.</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">Pesagens Recentes</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-gray-500">
              <th className="pb-2 pr-4">ID</th>
              <th className="pb-2 pr-4">Peso</th>
              <th className="pb-2 pr-4">Confianca</th>
              <th className="pb-2 pr-4">Data</th>
              <th className="pb-2">Evidencia</th>
            </tr>
          </thead>
          <tbody>
            {weighings.map((w) => (
              <tr key={w.id} className="border-b last:border-0">
                <td className="py-2 pr-4 font-medium">{w.id}</td>
                <td className="py-2 pr-4">{w.weight_kg.toFixed(1)} kg</td>
                <td className="py-2 pr-4">{w.confidence ? `${(w.confidence * 100).toFixed(0)}%` : '—'}</td>
                <td className="py-2 pr-4">{new Date(w.created_at).toLocaleString('pt-BR')}</td>
                <td className="py-2">{w.evidence_path ? <span className="text-green-600">Ver</span> : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
