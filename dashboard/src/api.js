const API_URL = import.meta.env.VITE_API_URL || '/api';

async function fetchJSON(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export async function getSummary() {
  return fetchJSON('/reports/summary');
}

export async function getWeighings(limit = 20) {
  return fetchJSON(`/weighings?limit=${limit}`);
}

export async function getAlerts() {
  return fetchJSON('/reports/alerts');
}

export async function getWeightHistory(days = 30) {
  return fetchJSON(`/reports/weight-history?days=${days}`);
}
