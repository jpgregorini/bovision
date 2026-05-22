export default function AlertsList({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">Alertas</h2>
        <p className="text-gray-400">Nenhum alerta ativo.</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">Alertas</h2>
      <ul className="space-y-3">
        {alerts.map((alert) => (
          <li key={alert.animal_id} className="flex items-start gap-3 rounded-lg bg-red-50 p-3">
            <span className="text-lg">&#9888;</span>
            <div>
              <p className="font-medium text-red-800">Animal #{alert.rfid}</p>
              <p className="text-sm text-red-600">
                Perda de {Math.abs(alert.change_percent).toFixed(1)}%
                ({(alert.current_weight_kg - alert.previous_weight_kg).toFixed(0)} kg)
                — de {alert.previous_weight_kg.toFixed(0)} kg para {alert.current_weight_kg.toFixed(0)} kg
              </p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
