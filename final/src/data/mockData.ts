// ─────────────────────────────────────────────────────────────────────────────
// NovaRoute.AI  —  Mock / Demo data
//
// ALL mock data is defined here.  The exported getMock* functions return data
// in exactly the same shape as the canonical API types (src/types/api.ts).
// Swapping mock → live backend requires no component changes; just pass
// useMock=false to the api.ts functions.
// ─────────────────────────────────────────────────────────────────────────────

import type {
  Junction, Officer, OfficerAllocation, Route,
  TrafficData, CoverageData, BaselineData,
  DeploymentResult, SimulateIncidentRequest,
} from '../types/api';

// ── Raw mock records ──────────────────────────────────────────────────────────
// Kept in their natural shape first, then projected to API types by the
// getMock* helpers below so nothing has to change when the backend is wired in.

const _junctions: Junction[] = [
  {
    id: 'J001', name: 'Sitabuldi Intersection', shortName: 'Sitabuldi',
    latitude: 21.1498, longitude: 79.0806, riskScore: 78, riskLevel: 'HIGH',
    factors: { accident: 28, traffic: 22, pedestrian: 18, timeOfDay: 10 },
    congestionFactor: 1.54, assignedOfficer: 'OFF001',
  },
  {
    id: 'J002', name: 'Itwari Station Square', shortName: 'Itwari',
    latitude: 21.1530, longitude: 79.0731, riskScore: 88, riskLevel: 'CRITICAL',
    factors: { accident: 35, traffic: 25, pedestrian: 20, timeOfDay: 8 },
    congestionFactor: 1.76, assignedOfficer: 'OFF002',
  },
  {
    id: 'J003', name: 'Wardha Road – Katol Crossing', shortName: 'Wardha–Katol',
    latitude: 21.1290, longitude: 79.1053, riskScore: 76, riskLevel: 'HIGH',
    factors: { accident: 26, traffic: 24, pedestrian: 16, timeOfDay: 10 },
    congestionFactor: 1.48, assignedOfficer: 'OFF003',
  },
  {
    id: 'J004', name: 'Kamptee Road Junction', shortName: 'Kamptee Rd',
    latitude: 21.1648, longitude: 79.1048, riskScore: 65, riskLevel: 'HIGH',
    factors: { accident: 22, traffic: 20, pedestrian: 14, timeOfDay: 9 },
    congestionFactor: 1.32, assignedOfficer: 'OFF004',
  },
  {
    id: 'J005', name: 'Subhash Road – Ganesh Peth', shortName: 'Subhash Rd',
    latitude: 21.1469, longitude: 79.0775, riskScore: 45, riskLevel: 'MEDIUM',
    factors: { accident: 14, traffic: 16, pedestrian: 10, timeOfDay: 5 },
    congestionFactor: 1.14, assignedOfficer: 'OFF005',
  },
  {
    id: 'J006', name: 'Hingna Road Junction', shortName: 'Hingna Rd',
    latitude: 21.0996, longitude: 79.0403, riskScore: 52, riskLevel: 'MEDIUM',
    factors: { accident: 18, traffic: 14, pedestrian: 12, timeOfDay: 8 },
    congestionFactor: 1.22, assignedOfficer: 'OFF006',
  },
  {
    id: 'J007', name: 'Wardha Road Junction', shortName: 'Wardha Rd',
    latitude: 21.1458, longitude: 79.0882, riskScore: 92, riskLevel: 'CRITICAL',
    factors: { accident: 32, traffic: 27, pedestrian: 21, timeOfDay: 12 },
    congestionFactor: 1.82, assignedOfficer: 'OFF007',
  },
  {
    id: 'J008', name: 'Amravati Road Junction', shortName: 'Amravati Rd',
    latitude: 21.1665, longitude: 79.0627, riskScore: 55, riskLevel: 'MEDIUM',
    factors: { accident: 18, traffic: 16, pedestrian: 13, timeOfDay: 8 },
    congestionFactor: 1.24, assignedOfficer: 'OFF008',
  },
  {
    id: 'J009', name: 'Narendra Nagar Square', shortName: 'Narendra Ngr',
    latitude: 21.1358, longitude: 79.0650, riskScore: 38, riskLevel: 'MEDIUM',
    factors: { accident: 12, traffic: 12, pedestrian: 8, timeOfDay: 6 },
    congestionFactor: 1.08, assignedOfficer: null,
  },
  {
    id: 'J010', name: 'MIDC Hingna Chowk', shortName: 'MIDC Hingna',
    latitude: 21.1055, longitude: 79.0255, riskScore: 28, riskLevel: 'LOW',
    factors: { accident: 8, traffic: 8, pedestrian: 6, timeOfDay: 6 },
    congestionFactor: 1.04, assignedOfficer: null,
  },
  {
    id: 'J011', name: 'Airport Road Junction', shortName: 'Airport Rd',
    latitude: 21.0924, longitude: 79.0567, riskScore: 43, riskLevel: 'MEDIUM',
    factors: { accident: 14, traffic: 14, pedestrian: 9, timeOfDay: 6 },
    congestionFactor: 1.12, assignedOfficer: null,
  },
  {
    id: 'J012', name: 'Koradi Road Crossing', shortName: 'Koradi Rd',
    latitude: 21.1793, longitude: 79.1063, riskScore: 67, riskLevel: 'HIGH',
    factors: { accident: 24, traffic: 20, pedestrian: 15, timeOfDay: 8 },
    congestionFactor: 1.36, assignedOfficer: 'OFF009',
  },
  {
    id: 'J013', name: 'Umrer Road Junction', shortName: 'Umrer Rd',
    latitude: 21.1116, longitude: 79.1032, riskScore: 58, riskLevel: 'MEDIUM',
    factors: { accident: 20, traffic: 16, pedestrian: 14, timeOfDay: 8 },
    congestionFactor: 1.26, assignedOfficer: null,
  },
  {
    id: 'J014', name: 'Ring Road – Mankapur', shortName: 'Mankapur',
    latitude: 21.1158, longitude: 79.0482, riskScore: 71, riskLevel: 'HIGH',
    factors: { accident: 26, traffic: 22, pedestrian: 14, timeOfDay: 9 },
    congestionFactor: 1.42, assignedOfficer: null,
  },
  {
    id: 'J015', name: 'Chhatrapati Square', shortName: 'Chhatrapati Sq',
    latitude: 21.1514, longitude: 79.0853, riskScore: 35, riskLevel: 'MEDIUM',
    factors: { accident: 10, traffic: 12, pedestrian: 8, timeOfDay: 5 },
    congestionFactor: 1.07, assignedOfficer: null,
  },
  {
    id: 'J016', name: 'Dharampeth T-Junction', shortName: 'Dharampeth',
    latitude: 21.1420, longitude: 79.0756, riskScore: 48, riskLevel: 'MEDIUM',
    factors: { accident: 16, traffic: 14, pedestrian: 11, timeOfDay: 7 },
    congestionFactor: 1.16, assignedOfficer: 'OFF010',
  },
  {
    id: 'J017', name: 'Cotton Market Square', shortName: 'Cotton Mkt',
    latitude: 21.1560, longitude: 79.0832, riskScore: 82, riskLevel: 'CRITICAL',
    factors: { accident: 30, traffic: 24, pedestrian: 20, timeOfDay: 8 },
    congestionFactor: 1.64, assignedOfficer: null,
  },
  {
    id: 'J018', name: 'VNIT T-Junction', shortName: 'VNIT',
    latitude: 21.1325, longitude: 79.0514, riskScore: 42, riskLevel: 'MEDIUM',
    factors: { accident: 14, traffic: 13, pedestrian: 9, timeOfDay: 6 },
    congestionFactor: 1.10, assignedOfficer: null,
  },
  {
    id: 'J019', name: 'Sadar Square', shortName: 'Sadar Sq',
    latitude: 21.1580, longitude: 79.0907, riskScore: 74, riskLevel: 'HIGH',
    factors: { accident: 26, traffic: 22, pedestrian: 16, timeOfDay: 10 },
    congestionFactor: 1.45, assignedOfficer: null,
  },
  {
    id: 'J020', name: 'Gandhi Sagar Road', shortName: 'Gandhi Sgr',
    latitude: 21.1695, longitude: 79.0832, riskScore: 30, riskLevel: 'LOW',
    factors: { accident: 10, traffic: 10, pedestrian: 6, timeOfDay: 4 },
    congestionFactor: 1.05, assignedOfficer: null,
  },
];

const _officers: Officer[] = [
  { id: 'OFF001', name: 'Officer Sharma',   badge: 'OFF-01', latitude: 21.1510, longitude: 79.0790, status: 'AVAILABLE', locationName: 'Sitabuldi',       assignedJunction: 'J001', responseTime: 3.8, distanceKm: 2.9, locked: false },
  { id: 'OFF002', name: 'Officer Patel',    badge: 'OFF-02', latitude: 21.1555, longitude: 79.0710, status: 'AVAILABLE', locationName: 'Itwari Depot',     assignedJunction: 'J002', responseTime: 4.1, distanceKm: 3.2, locked: false },
  { id: 'OFF003', name: 'Officer Deshmukh', badge: 'OFF-03', latitude: 21.1270, longitude: 79.1070, status: 'ASSIGNED',  locationName: 'Wardha Road Beat', assignedJunction: 'J003', responseTime: 3.6, distanceKm: 2.7, locked: false },
  { id: 'OFF004', name: 'Officer Kulkarni', badge: 'OFF-04', latitude: 21.1670, longitude: 79.1030, status: 'LOCKED',    locationName: 'Kamptee Outpost',  assignedJunction: 'J004', responseTime: 6.0, distanceKm: 4.8, locked: true  },
  { id: 'OFF005', name: 'Officer Thakur',   badge: 'OFF-05', latitude: 21.1488, longitude: 79.0760, status: 'AVAILABLE', locationName: 'Ganesh Peth',      assignedJunction: 'J005', responseTime: 2.9, distanceKm: 2.2, locked: false },
  { id: 'OFF006', name: 'Officer Mishra',   badge: 'OFF-06', latitude: 21.1020, longitude: 79.0420, status: 'AVAILABLE', locationName: 'Hingna Post',      assignedJunction: 'J006', responseTime: 4.5, distanceKm: 3.5, locked: false },
  { id: 'OFF007', name: 'Officer Joshi',    badge: 'OFF-07', latitude: 21.1440, longitude: 79.0860, status: 'ASSIGNED',  locationName: 'Wardha Junction',  assignedJunction: 'J007', responseTime: 4.2, distanceKm: 3.7, locked: false },
  { id: 'OFF008', name: 'Officer Yadav',    badge: 'OFF-08', latitude: 21.1680, longitude: 79.0610, status: 'AVAILABLE', locationName: 'Amravati Circle',  assignedJunction: 'J008', responseTime: 3.2, distanceKm: 2.4, locked: false },
  { id: 'OFF009', name: 'Officer Bane',     badge: 'OFF-09', latitude: 21.1800, longitude: 79.1045, status: 'ASSIGNED',  locationName: 'Koradi Outpost',   assignedJunction: 'J012', responseTime: 5.1, distanceKm: 4.1, locked: false },
  { id: 'OFF010', name: 'Officer Singh',    badge: 'OFF-10', latitude: 21.1435, longitude: 79.0740, status: 'AVAILABLE', locationName: 'Dharampeth Beat',  assignedJunction: 'J016', responseTime: 3.4, distanceKm: 2.6, locked: false },
  { id: 'OFF011', name: 'Officer Wagh',     badge: 'OFF-11', latitude: 21.1600, longitude: 79.0950, status: 'OFFLINE',   locationName: 'Control Room',     assignedJunction: null,   responseTime: null, distanceKm: null, locked: false },
  { id: 'OFF012', name: 'Officer Naik',     badge: 'OFF-12', latitude: 21.1380, longitude: 79.0900, status: 'OFFLINE',   locationName: 'Control Room',     assignedJunction: null,   responseTime: null, distanceKm: null, locked: false },
];

const _allocations: OfficerAllocation[] = [
  {
    junctionId: 'J007', officerId: 'OFF007', responseTime: 4.2, distanceKm: 3.7,
    alternatives: [
      { officerId: 'OFF001', responseTime: 7.1 },
      { officerId: 'OFF010', responseTime: 6.4 },
      { officerId: 'OFF005', responseTime: 8.3 },
    ],
    reason: 'Officer Joshi (OFF-07) has the shortest Dijkstra-computed response time to this high-risk junction via the Wardha Road corridor.',
  },
  {
    junctionId: 'J002', officerId: 'OFF002', responseTime: 4.1, distanceKm: 3.2,
    alternatives: [
      { officerId: 'OFF001', responseTime: 5.8 },
      { officerId: 'OFF005', responseTime: 7.2 },
    ],
    reason: 'Officer Patel (OFF-02) is stationed nearest to Itwari with optimal road-network routing.',
  },
  {
    junctionId: 'J001', officerId: 'OFF001', responseTime: 3.8, distanceKm: 2.9,
    alternatives: [
      { officerId: 'OFF005', responseTime: 4.6 },
      { officerId: 'OFF010', responseTime: 5.1 },
    ],
    reason: 'Officer Sharma (OFF-01) is the closest available unit to Sitabuldi Intersection.',
  },
];

const _routes: Route[] = [
  {
    officerId: 'OFF007', junctionId: 'J007', responseTime: 4.2, distance: 3.7,
    path: [[21.1440, 79.0860], [21.1445, 79.0865], [21.1450, 79.0872], [21.1458, 79.0882]],
  },
  {
    officerId: 'OFF001', junctionId: 'J001', responseTime: 3.8, distance: 2.9,
    path: [[21.1510, 79.0790], [21.1505, 79.0798], [21.1500, 79.0803], [21.1498, 79.0806]],
  },
  {
    officerId: 'OFF002', junctionId: 'J002', responseTime: 4.1, distance: 3.2,
    path: [[21.1555, 79.0710], [21.1545, 79.0718], [21.1537, 79.0725], [21.1530, 79.0731]],
  },
  {
    officerId: 'OFF003', junctionId: 'J003', responseTime: 3.6, distance: 2.7,
    path: [[21.1270, 79.1070], [21.1275, 79.1065], [21.1282, 79.1058], [21.1290, 79.1053]],
  },
  {
    officerId: 'OFF009', junctionId: 'J012', responseTime: 5.1, distance: 4.1,
    path: [[21.1800, 79.1045], [21.1798, 79.1050], [21.1796, 79.1057], [21.1793, 79.1063]],
  },
];

const _traffic: TrafficData = {
  vehicleCount: 1284,
  avgDensity: 0.64,
  congestionLevel: 'MODERATE',
  congestionFactor: 1.24,
  networkStatus: 'LIVE',
  timestamp: '10:42:31',
  trend: [0.52, 0.55, 0.58, 0.60, 0.61, 0.63, 0.64, 0.63, 0.65, 0.64],
};

const _coverage: CoverageData = {
  coveragePct: 85,
  coveredJunctions: 17,
  totalJunctions: 20,
  thresholdMinutes: 6,
  avgResponse: 4.2,
  worstResponse: 8.1,
  uncoveredCount: 3,
  uncoveredJunctions: ['J014', 'J019', 'J017'],
};

const _baseline: BaselineData = {
  static:    { avgResponse: 8.4, coveragePct: 48, coveredCount: 10, total: 20 },
  novaroute: { avgResponse: 4.2, coveragePct: 85, coveredCount: 17, total: 20 },
};

// ── Public getter functions (used by api.ts) ──────────────────────────────────
// Return fresh copies so state mutations in App don't affect the mock source.

export const getMockJunctions    = (): Junction[]          => _junctions.map(j => ({ ...j }));
export const getMockOfficers     = (): Officer[]           => _officers.map(o => ({ ...o }));
export const getMockAllocations  = (): OfficerAllocation[] => _allocations.map(a => ({ ...a, alternatives: [...a.alternatives] }));
export const getMockRoutes       = (): Route[]             => _routes.map(r => ({ ...r, path: [...r.path] }));
export const getMockTraffic      = (): TrafficData         => ({ ..._traffic, trend: [..._traffic.trend] });
export const getMockCoverage     = (): CoverageData        => ({ ..._coverage, uncoveredJunctions: [..._coverage.uncoveredJunctions] });
export const getMockBaseline     = (): BaselineData        => ({ static: { ..._baseline.static }, novaroute: { ..._baseline.novaroute } });

// ── Refreshed mock data — applies small random variation so demo refreshes are visible ──
// Vary a number by ±pct% (default ±8%), clamped to [min, max].
function vary(value: number, pct = 0.08, min = 0, max = Infinity): number {
  const delta = value * pct * (Math.random() * 2 - 1);
  return Math.max(min, Math.min(max, Math.round((value + delta) * 10) / 10));
}

export function getRefreshedMockJunctions(): Junction[] {
  return _junctions.map(j => {
    const newScore = vary(j.riskScore, 0.06, 1, 100);
    const level: Junction['riskLevel'] =
      newScore >= 76 ? 'CRITICAL' : newScore >= 56 ? 'HIGH' : newScore >= 31 ? 'MEDIUM' : 'LOW';
    return {
      ...j,
      riskScore: newScore,
      riskLevel: level,
      congestionFactor: vary(j.congestionFactor, 0.08, 0.5, 3.5),
      factors: {
        accident:   vary(j.factors.accident,   0.07, 0),
        traffic:    vary(j.factors.traffic,    0.07, 0),
        pedestrian: vary(j.factors.pedestrian, 0.07, 0),
        timeOfDay:  vary(j.factors.timeOfDay,  0.07, 0),
      },
    };
  });
}

export function getRefreshedMockOfficers(): Officer[] {
  return _officers.map(o => ({
    ...o,
    responseTime: o.responseTime != null ? vary(o.responseTime, 0.1, 1, 20) : null,
  }));
}

export function getRefreshedMockTraffic(): TrafficData {
  return {
    ..._traffic,
    vehicleCount: vary(_traffic.vehicleCount, 0.07, 0),
    avgDensity:   vary(_traffic.avgDensity,   0.08, 0),
    congestionFactor: vary(_traffic.congestionFactor, 0.06, 0.5, 3),
    trend: _traffic.trend.map(v => vary(v, 0.12, 0)),
    timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
  };
}

export function getRefreshedMockCoverage(): CoverageData {
  const coveredJunctions = vary(_coverage.coveredJunctions, 0.05, 10, _coverage.totalJunctions);
  const coveragePct = Math.round((coveredJunctions / _coverage.totalJunctions) * 100);
  return {
    ..._coverage,
    coveredJunctions,
    coveragePct,
    avgResponse: vary(_coverage.avgResponse, 0.08, 1, 10),
    worstResponse: vary(_coverage.worstResponse, 0.08, 3, 15),
  };
}

export function getMockRoute(officerId: string, junctionId: string): Route | null {
  return _routes.find(r => r.officerId === officerId && r.junctionId === junctionId) ?? null;
}

export function buildMockDeploymentResult(payload: SimulateIncidentRequest): DeploymentResult {
  const junc = _junctions.find(j => j.id === payload.junctionId);
  const prev = junc?.riskScore ?? 72;
  const newScore = Math.min(100, prev + 18 + Math.floor(Math.random() * 8));
  const prevCongestion = junc?.congestionFactor ?? 1.12;
  const newCongestion  = prevCongestion + 1.1 + Math.random() * 0.3;
  const timeBefore = 8.4;
  const timeAfter  = 4.2;
  return {
    affectedJunction: payload.junctionId,
    prevRiskScore: prev,
    newRiskScore: newScore,
    prevCongestion,
    newCongestion,
    previousOfficers: [
      { officerId: 'OFF001', fromJunction: 'J001',            toJunction: 'J001' },
      { officerId: 'OFF007', fromJunction: payload.junctionId, toJunction: payload.junctionId },
    ],
    newOfficers: [
      { officerId: 'OFF001', fromJunction: 'J001',            toJunction: payload.junctionId },
      { officerId: 'OFF007', fromJunction: payload.junctionId, toJunction: 'J001' },
    ],
    responseTimeBefore: timeBefore,
    responseTimeAfter:  timeAfter,
    improvementPercentage: Math.round((1 - timeAfter / timeBefore) * 100),
  };
}

// ── Legacy re-exports for any code that still imports from this file directly ─
// These keep backward compatibility while the codebase migrates to api.ts.
// Remove these once all consumers go through the service layer.
export type { Junction, Officer, OfficerAllocation as Allocation, Route, TrafficData, CoverageData, BaselineData };
