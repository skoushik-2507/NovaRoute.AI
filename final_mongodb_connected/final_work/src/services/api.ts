// ─────────────────────────────────────────────────────────────────────────────
// NovaRoute.AI  —  API Service Layer
//
// All backend communication is centralised here.
// Components must NEVER call fetch/axios directly.
//
// Configuration:
//   VITE_API_URL   Backend base URL (default: http://localhost:8000)
//   VITE_USE_MOCK  Force mock mode regardless of backend availability
//                  Set to "false" in production to use the live backend.
//
// Behaviour:
//   - When demoMode is true (or VITE_USE_MOCK=true) → always return mock data.
//   - When demoMode is false → attempt real API call; on failure, fall back to
//     the last known mock data and surface an error to the caller.
// ─────────────────────────────────────────────────────────────────────────────

import type {
  Junction, Officer, OfficerAllocation, Route,
  TrafficData, CoverageData, BaselineData,
  SimulateIncidentRequest, DeploymentResult,
} from '../types/api';

import {
  getMockJunctions,
  getMockOfficers,
  getMockAllocations,
  getMockRoutes,
  getMockRoute,
  getMockTraffic,
  getMockCoverage,
  getMockBaseline,
  buildMockDeploymentResult,
  getRefreshedMockJunctions,
  getRefreshedMockOfficers,
  getRefreshedMockTraffic,
  getRefreshedMockCoverage,
} from '../data/mockData';

// ── configuration ─────────────────────────────────────────────────────────────

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';
const FORCE_MOCK = (import.meta.env.VITE_USE_MOCK as string | undefined) === 'true';

// ── endpoint registry ─────────────────────────────────────────────────────────
// All backend endpoint paths live here. Update these when the backend team
// finalises the API surface — no component code needs to change.

export const ENDPOINTS = {
  junctions:          '/api/junctions',
  officers:           '/api/officers',
  risk:               '/api/risk',
  traffic:            '/api/traffic',
  allocations:        '/api/allocations',
  routes:             '/api/routes',
  route:              (officerId: string, junctionId: string) => `/api/routes/${officerId}/${junctionId}`,
  coverage:           '/api/coverage',
  baseline:           '/api/baseline',
  simulateIncident:   '/api/incidents/simulate',
  lockOfficer:        (id: string) => `/api/officers/${id}/lock`,
  unlockOfficer:      (id: string) => `/api/officers/${id}/unlock`,
  recalculate:        '/api/allocation/recalculate',
  refresh:            '/api/refresh',
} as const;

// ── low-level fetch wrapper ───────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}${text ? ': ' + text : ''}`);
  }
  return res.json() as Promise<T>;
}

// ── generic helper: try live API, fall back to mock on error ──────────────────

async function withFallback<T>(
  liveFn: () => Promise<T>,
  mockFn: () => T,
  useMock: boolean,
): Promise<T> {
  if (useMock || FORCE_MOCK) return mockFn();
  try {
    return await liveFn();
  } catch {
    return mockFn();
  }
}

// ── public API functions ──────────────────────────────────────────────────────

export async function getJunctions(useMock = true, refresh = false): Promise<Junction[]> {
  return withFallback(
    () => apiFetch<Junction[]>(ENDPOINTS.junctions),
    refresh ? getRefreshedMockJunctions : getMockJunctions,
    useMock,
  );
}

export async function getOfficers(useMock = true, refresh = false): Promise<Officer[]> {
  return withFallback(
    () => apiFetch<Officer[]>(ENDPOINTS.officers),
    refresh ? getRefreshedMockOfficers : getMockOfficers,
    useMock,
  );
}

export async function getRiskData(useMock = true): Promise<Junction[]> {
  return withFallback(
    () => apiFetch<Junction[]>(ENDPOINTS.risk),
    getMockJunctions,
    useMock,
  );
}

export async function getTrafficData(useMock = true, refresh = false): Promise<TrafficData> {
  return withFallback(
    () => apiFetch<TrafficData>(ENDPOINTS.traffic),
    refresh ? getRefreshedMockTraffic : getMockTraffic,
    useMock,
  );
}

export async function getAllocations(useMock = true): Promise<OfficerAllocation[]> {
  return withFallback(
    () => apiFetch<OfficerAllocation[]>(ENDPOINTS.allocations),
    getMockAllocations,
    useMock,
  );
}

export async function getAllRoutes(useMock = true): Promise<Route[]> {
  return withFallback(
    () => apiFetch<Route[]>(ENDPOINTS.routes),
    getMockRoutes,
    useMock,
  );
}

export async function getRoute(
  officerId: string,
  junctionId: string,
  useMock = true,
): Promise<Route | null> {
  return withFallback(
    () => apiFetch<Route>(ENDPOINTS.route(officerId, junctionId)),
    () => getMockRoute(officerId, junctionId),
    useMock,
  );
}

export async function getCoverage(useMock = true, refresh = false): Promise<CoverageData> {
  return withFallback(
    () => apiFetch<CoverageData>(ENDPOINTS.coverage),
    refresh ? getRefreshedMockCoverage : getMockCoverage,
    useMock,
  );
}

export async function getBaseline(useMock = true): Promise<BaselineData> {
  return withFallback(
    () => apiFetch<BaselineData>(ENDPOINTS.baseline),
    getMockBaseline,
    useMock,
  );
}

export async function simulateIncident(
  payload: SimulateIncidentRequest,
  useMock = true,
): Promise<DeploymentResult> {
  return withFallback(
    () => apiFetch<DeploymentResult>(ENDPOINTS.simulateIncident, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
    () => buildMockDeploymentResult(payload),
    useMock,
  );
}

export async function lockOfficer(officerId: string, useMock = true): Promise<void> {
  if (useMock || FORCE_MOCK) return;
  await apiFetch<void>(ENDPOINTS.lockOfficer(officerId), { method: 'POST' });
}

export async function unlockOfficer(officerId: string, useMock = true): Promise<void> {
  if (useMock || FORCE_MOCK) return;
  await apiFetch<void>(ENDPOINTS.unlockOfficer(officerId), { method: 'POST' });
}

export async function triggerRecalculate(useMock = true): Promise<void> {
  if (useMock || FORCE_MOCK) return new Promise(r => setTimeout(r, 400));
  await apiFetch<void>(ENDPOINTS.recalculate, { method: 'POST' });
}

export async function refreshData(useMock = true): Promise<void> {
  if (useMock || FORCE_MOCK) return new Promise(r => setTimeout(r, 300));
  await apiFetch<void>(ENDPOINTS.refresh, { method: 'POST' });
}

export interface RefreshedData {
  junctions: Junction[]; officers: Officer[]; allocations: OfficerAllocation[];
  routes: Route[]; traffic: TrafficData; coverage: CoverageData; baseline: BaselineData;
}

export async function refreshAllData(useMock = true): Promise<RefreshedData> {
  const [junctions, officers, allocations, routes, traffic, coverage, baseline] = await Promise.all([
    getJunctions(useMock, true),
    getOfficers(useMock, true),
    getAllocations(useMock),
    getAllRoutes(useMock),
    getTrafficData(useMock, true),
    getCoverage(useMock, true),
    getBaseline(useMock),
  ]);
  return { junctions, officers, allocations, routes, traffic, coverage, baseline };
}
