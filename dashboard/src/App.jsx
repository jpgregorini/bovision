import { useEffect, useState } from 'react';
import * as api from './api';
import StatsCards from './components/StatsCards';
import WeightChart from './components/WeightChart';
import WeighingsTable from './components/WeighingsTable';
import AlertsList from './components/AlertsList';

const REFRESH_INTERVAL = 30_000;

export default function App() {
  const [summary, setSummary] = useState(null);
  const [weighings, setWeighings] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  async function fetchAll() {
    try {
      const [s, w, a, h] = await Promise.all([
        api.getSummary(), api.getWeighings(), api.getAlerts(), api.getWeightHistory(),
      ]);
      setSummary(s); setWeighings(w); setAlerts(a); setHistory(h); setError(null);
    } catch (err) {
      setError('Falha ao conectar com a API. Verifique se o servidor esta rodando.');
      console.error(err);
    }
  }

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <h1 className="text-2xl font-bold text-green-700">BoviSight</h1>
          <p className="text-sm text-gray-500">Pesagem automatizada por visao computacional</p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6">
        {error && <div className="rounded-lg bg-yellow-50 p-4 text-yellow-800">{error}</div>}
        <StatsCards summary={summary} />
        <WeightChart data={history} />
        <WeighingsTable weighings={weighings} />
        <AlertsList alerts={alerts} />
      </main>
    </div>
  );
}
