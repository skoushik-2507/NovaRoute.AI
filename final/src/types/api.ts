// ─────────────────────────────────────────────────────────────────────────────
// NovaRoute.AI  —  Canonical backend API types
//
// These types mirror the exact shapes the FastAPI backend will return.
// All frontend components must consume these types (or their adapted equivalents).
// The mock data layer in src/data/mockData.ts adapts its own shapes to these
// types via the toAPI* helpers so the UI never needs to change when the real
// backend is connected.
// ─────────────────────────────────────────────────────────────────────────────

// ── Risk ─────────────────────────────────────────────────────────────────────

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface RiskFactors {
  accident: number;
  traffic: number;
  pedestrian: number;
  timeOfDay: number;
}

export interface RiskScore {
  junctionId: string;
  score: number;       // 0-100
  level: RiskLevel;
  factors: RiskFactors;
  congestionFactor: number;
  computedAt: string;  // ISO timestamp
}

// ── Junction ──────────────────────────────────────────────────────────────────

export interface Junction {
  id: string;
  name: string;
  shortName: string;
  latitude: number;
  longitude: number;
  riskScore: number;
  riskLevel: RiskLevel;
  factors: RiskFactors;
  congestionFactor: number;
  assignedOfficer: string | null;
}

// ── Officer ───────────────────────────────────────────────────────────────────

export type OfficerStatus = 'AVAILABLE' | 'ASSIGNED' | 'LOCKED' | 'OFFLINE';

export interface Officer {
  id: string;
  name: string;
  badge: string;
  latitude: number;
  longitude: number;
  status: OfficerStatus;
  locationName: string;
  assignedJunction: string | null;
  responseTime: number | null;   // minutes
  distanceKm: number | null;
  locked: boolean;
}

// ── Allocation ────────────────────────────────────────────────────────────────

export interface AllocationAlternative {
  officerId: string;
  responseTime: number;
}

export interface OfficerAllocation {
  junctionId: string;
  officerId: string;
  responseTime: number;
  distanceKm: number;
  alternatives: AllocationAlternative[];
  reason: string;
}

// ── Route (Dijkstra path) ─────────────────────────────────────────────────────

export interface Route {
  officerId: string;
  junctionId: string;
  responseTime: number;
  distance: number;            // km
  path: [number, number][];   // [latitude, longitude] pairs
}

// ── Traffic ───────────────────────────────────────────────────────────────────

export type CongestionLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'SEVERE';
export type NetworkStatus = 'LIVE' | 'OFFLINE';

export interface TrafficData {
  junctionId?: string;         // optional: null = network-wide aggregate
  vehicleCount: number;
  avgDensity: number;
  congestionLevel: CongestionLevel;
  congestionFactor: number;
  networkStatus: NetworkStatus;
  timestamp: string;           // ISO timestamp or human-readable time
  trend: number[];             // recent density readings for sparkline
}

// ── Incident ──────────────────────────────────────────────────────────────────

export type IncidentType = 'Accident' | 'Breakdown' | 'Protest' | 'VIP Movement' | 'Medical Emergency';
export type IncidentSeverity = 'Low' | 'Medium' | 'High' | 'Critical';
export type IncidentStatus = 'ACTIVE' | 'RESOLVED' | 'PENDING';

export interface Incident {
  id: string;
  junctionId: string;
  type: IncidentType;
  severity: IncidentSeverity;
  timestamp: string;
  status: IncidentStatus;
  congestionImpact: boolean;
}

// ── Simulate incident request / response ─────────────────────────────────────

export interface SimulateIncidentRequest {
  junctionId: string;
  incidentType: IncidentType | string;
  severity: IncidentSeverity | string;
  congestionImpact: boolean;
}

export interface DeploymentChange {
  officerId: string;
  fromJunction: string | null;
  toJunction: string;
}

export interface DeploymentResult {
  affectedJunction: string;
  prevRiskScore: number;
  newRiskScore: number;
  prevCongestion: number;
  newCongestion: number;
  previousOfficers: DeploymentChange[];
  newOfficers: DeploymentChange[];
  responseTimeBefore: number;
  responseTimeAfter: number;
  improvementPercentage: number;
}

// ── Coverage ──────────────────────────────────────────────────────────────────

export interface CoverageData {
  coveragePct: number;
  coveredJunctions: number;
  totalJunctions: number;
  thresholdMinutes: number;
  avgResponse: number;
  worstResponse: number;
  uncoveredCount: number;
  uncoveredJunctions: string[];
}

// ── Baseline comparison ───────────────────────────────────────────────────────

export interface BaselineScenario {
  avgResponse: number;
  coveragePct: number;
  coveredCount: number;
  total: number;
}

export interface BaselineData {
  static: BaselineScenario;
  novaroute: BaselineScenario;
}

// ── API response wrappers ─────────────────────────────────────────────────────

export type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T; fetchedAt: number }
  | { status: 'error'; message: string; lastData?: T };
