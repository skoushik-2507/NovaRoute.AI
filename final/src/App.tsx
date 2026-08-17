import { useState, useEffect, useCallback, useRef } from 'react';
import {
  getMockJunctions, getMockOfficers, getMockAllocations,
  getMockRoutes, getMockTraffic, getMockCoverage, getMockBaseline,
} from './data/mockData';

// Inline SVG icon components — no external dependency
type IconProps = { size?: number; style?: React.CSSProperties; className?: string };
const ic = (path: string) => ({ size = 16, style, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={style} className={className}>{path.split('|').map((d, i) => d.startsWith('circle') ? <circle key={i} cx={d.split(',')[1]} cy={d.split(',')[2]} r={d.split(',')[3]} /> : <path key={i} d={d} />)}</svg>
);
const AlertTriangle = ic('M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z|M12 9v4|M12 17h.01');
const Activity = ic('M22 12h-4l-3 9L9 3l-3 9H2');
const Users = ic('M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2|circle,9,7,4|M23 21v-2a4 4 0 00-3-3.87|M16 3.13a4 4 0 010 7.75');
const MapPin = ic('M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z|circle,12,10,3');
const Shield = ic('M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z');
const BarChart2 = ic('M18 20V10|M12 20V4|M6 20v-6');
const Radio = ic('M4.9 19.1C1 15.2 1 8.8 4.9 4.9|M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5|circle,12,12,2|M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5|M19.1 4.9C23 8.8 23 15.1 19.1 19');
const RefreshCw = ic('M23 4v6h-6|M1 20v-6h6|M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15');
const ChevronDown = ic('M6 9l6 6 6-6');
const Lock = ic('M19 11H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2z|M7 11V7a5 5 0 0110 0v4');
const Unlock = ic('M19 11H5a2 2 0 00-2 2v7a2 2 0 002 2h14a2 2 0 002-2v-7a2 2 0 00-2-2z|M7 11V7a5 5 0 019.9-1');
const X = ic('M18 6L6 18|M6 6l12 12');
const CheckCircle = ic('M22 11.08V12a10 10 0 11-5.93-9.14|M22 4L12 14.01l-3-3');
const AlertCircle = ic('circle,12,12,10|M12 8v4|M12 16h.01');
const TrendingDown = ic('M23 18l-9.5-9.5-5 5L1 6|M17 18h6v-6');
const Eye = ic('M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z|circle,12,12,3');
const EyeOff = ic('M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24|M1 1l22 22');
const Menu = ic('M3 12h18|M3 6h18|M3 18h18');
const Camera = ic('M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z|circle,12,13,4');
const Zap = ic('M13 2L3 14h9l-1 8 10-12h-9l1-8z');
import type { Junction, Officer, Route, OfficerAllocation, DeploymentResult } from './types/api';
import { simulateIncident, lockOfficer, unlockOfficer, refreshAllData } from './services/api';
import { useNovaRouteData } from './hooks/useNovaRouteData';
import MapView, { type LayerState } from './components/MapView';

// ─── color helpers ──────────────────────────────────────────────────────────
const RISK_COLOR: Record<string, string> = { LOW: '#00c896', MEDIUM: '#ffd700', HIGH: '#ff8c42', CRITICAL: '#ff3a3a' };
const RISK_BG: Record<string, string> = { LOW: 'rgba(0,200,150,0.12)', MEDIUM: 'rgba(255,215,0,0.1)', HIGH: 'rgba(255,140,66,0.12)', CRITICAL: 'rgba(255,58,58,0.12)' };
const STATUS_COLOR: Record<string, string> = { AVAILABLE: '#00d084', ASSIGNED: '#f59e0b', LOCKED: '#6b7280', OFFLINE: '#374151' };

// ─── filter types + defaults ─────────────────────────────────────────────────
interface FilterState { time: string; risk: string; traffic: string; officers: string; area: string; }
const DEFAULT_FILTERS: FilterState = { time: 'Live', risk: 'All', traffic: 'All', officers: 'All', area: 'Nagpur City' };

const AREA_KEYWORDS: Record<string, string> = {
  'Wardha Rd': 'Wardha', 'Itwari': 'Itwari', 'Cotton Market': 'Cotton',
  'Sitabuldi': 'Sitabuldi', 'Sadar': 'Sadar', 'Mankapur': 'Mankapur', 'Koradi Rd': 'Koradi',
};
const OFFICER_STATUS_MAP: Record<string, string> = {
  'Available': 'AVAILABLE', 'Deployed': 'ASSIGNED', 'Locked': 'LOCKED', 'Offline': 'OFFLINE',
};
const CONGESTION_BANDS: Record<string, (cf: number) => boolean> = {
  'Low':      cf => cf < 1.2,
  'Moderate': cf => cf >= 1.2 && cf < 1.5,
  'High':     cf => cf >= 1.5 && cf < 2.0,
  'Severe':   cf => cf >= 2.0,
};

// ─── tiny components ─────────────────────────────────────────────────────────
const PulseDot = ({ color = '#00d084' }: { color?: string }) => (
  <span className="live-dot" style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0 }} />
);

const Badge = ({ label, color, bg }: { label: string; color: string; bg: string }) => (
  <span style={{ background: bg, color, border: `1px solid ${color}40`, borderRadius: 2, padding: '1px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.04em' }}>{label}</span>
);

const RiskBadge = ({ level }: { level: string }) => (
  <Badge label={level} color={RISK_COLOR[level]} bg={RISK_BG[level]} />
);

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <div style={{ color: '#4a6080', fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>{children}</div>
);

const Divider = () => <div style={{ height: 1, background: '#1c3050', margin: '10px 0' }} />;

const MiniBar = ({ value, max, color }: { value: number; max: number; color: string }) => (
  <div style={{ flex: 1, height: 4, background: '#1c3050', borderRadius: 2, overflow: 'hidden' }}>
    <div style={{ width: `${(value / max) * 100}%`, height: '100%', background: color, borderRadius: 2 }} />
  </div>
);

// ─── loading skeletons (must be before App so they're defined when used) ─────

function KPILoading() {
  return (
    <div style={{ display: 'flex', height: 76, background: '#080e1e', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
      {[1, 2, 3, 4, 5].map(i => (
        <div key={i} style={{
          flex: 1, borderRight: '1px solid #1c3050', padding: '10px 14px',
          display: 'flex', flexDirection: 'column', gap: 6, justifyContent: 'space-between',
        }}>
          <div style={{ height: 8, width: '60%', background: '#1c3050', borderRadius: 2 }} />
          <div style={{ height: 20, width: '40%', background: '#162240', borderRadius: 2 }} />
          <div style={{ height: 7, width: '50%', background: '#1c3050', borderRadius: 2 }} />
        </div>
      ))}
    </div>
  );
}

function PanelLoading({ label }: { label: string }) {
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#2a4060', fontSize: 11 }}>
        <RefreshCw size={11} />
        Loading…
      </div>
    </div>
  );
}

// ─── sidebar nav items ────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'dashboard', icon: Activity, label: 'Dashboard' },
  { id: 'risk', icon: AlertTriangle, label: 'Risk Intelligence' },
  { id: 'traffic', icon: Radio, label: 'Live Traffic' },
  { id: 'officers', icon: Shield, label: 'Officers' },
  { id: 'deployment', icon: MapPin, label: 'Deployment' },
  { id: 'incidents', icon: AlertCircle, label: 'Incidents' },
  { id: 'coverage', icon: Eye, label: 'Coverage' },
  { id: 'performance', icon: BarChart2, label: 'Performance' },
];

// ─── processing steps ─────────────────────────────────────────────────────────
const PROCESS_STEPS = [
  'Updating traffic conditions',
  'Updating graph weights',
  'Running Dijkstra algorithm',
  'Recalculating assignments',
  'Redeployment ready',
];

// ─── main app ─────────────────────────────────────────────────────────────────
export default function App() {
  // Demo mode drives whether api.ts uses mock data or the live backend
  const [demoMode, setDemoMode] = useState(true);

  // All data loaded via the service layer — components consume these directly
  const {
    data: { junctions: loadedJunctions, officers: loadedOfficers, routes: loadedRoutes,
            allocations: loadedAllocations, traffic: loadedTraffic,
            coverage: loadedCoverage, baseline: loadedBaseline },
    status: dataStatus,
    error: dataError,
    lastUpdated: dataLastUpdated,
    isBackendOnline,
    connectionMode,
    setJunctions,
    setOfficers,
    setRoutes: setRoutesHook,
    setTraffic: setTrafficHook,
    setCoverage: setCoverageHook,
    setBaseline: setBaselineHook,
    setAllocations: setAllocationsHook,
  } = useNovaRouteData(demoMode);

  // Local copies seeded synchronously so first render never has empty data
  const [junctions, setJunctionsLocal] = useState<Junction[]>(getMockJunctions);
  const [officers,  setOfficersLocal]  = useState<Officer[]>(getMockOfficers);

  // Sync from hook into local state once loaded
  useEffect(() => { if (loadedJunctions.length) setJunctionsLocal(loadedJunctions); }, [loadedJunctions]);
  useEffect(() => { if (loadedOfficers.length)  setOfficersLocal(loadedOfficers);  }, [loadedOfficers]);

  // Proxy setters so handlers update both local state and the hook's state
  const updateJunctions = (updater: (prev: Junction[]) => Junction[]) => {
    setJunctionsLocal(updater);
    setJunctions(updater);
  };
  const updateOfficers = (updater: (prev: Officer[]) => Officer[]) => {
    setOfficersLocal(updater);
    setOfficers(updater);
  };

  const routes      = loadedRoutes.length    ? loadedRoutes    : getMockRoutes();
  const coverage    = loadedCoverage         ?? getMockCoverage();
  const baseline    = loadedBaseline         ?? getMockBaseline();
  const traffic     = loadedTraffic          ?? getMockTraffic();
  const allocations = loadedAllocations.length ? loadedAllocations : getMockAllocations();

  // UI state
  const [selectedJunction, setSelectedJunction] = useState<Junction | null>(null);
  const [selectedOfficer, setSelectedOfficer] = useState<Officer | null>(null);
  const [activeRoute, setActiveRoute] = useState<Route | null>(null);
  const [showExplainability, setShowExplainability] = useState(false);
  const [showIncidentModal, setShowIncidentModal] = useState(false);
  const [incidentStep, setIncidentStep] = useState<'form' | 'processing' | 'result'>('form');
  const [processStep, setProcessStep] = useState(0);
  const [incidentResult, setIncidentResult] = useState<DeploymentResult | null>(null);
  const [incidentJunctionId, setIncidentJunctionId] = useState<string | null>(null);
  const [activeSidebar, setActiveSidebar] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('10:42:31 AM');
  const [refreshToast, setRefreshToast] = useState<'success' | 'error' | null>(null);
  const [timeline, setTimeline] = useState<{ time: string; label: string }[]>([]);

  const [layers, setLayers] = useState({
    risk: true, officers: true, routes: true, traffic: true, incidents: true,
  });
  const [incidentForm, setIncidentForm] = useState({
    junction: 'J007', type: 'Accident', severity: 'High', congestion: true,
  });

  // ─── filter state ───────────────────────────────────────────────────────────
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const setFilter = (key: keyof FilterState, value: string) =>
    setFilters(f => ({ ...f, [key]: value }));
  const resetFilters = () => setFilters(DEFAULT_FILTERS);

  // ─── filtered derived data ───────────────────────────────────────────────────
  // Time filter: for live/demo data, "Today" and shorter windows show all data.
  // "Last 15 min" / "Last 30 min" reduce visible junctions to higher-activity ones
  // (congestionFactor > 1.1) to simulate a narrower time slice without fabricating timestamps.
  const TIME_ACTIVITY_THRESHOLD: Record<string, number> = {
    'Live': 0, 'Last 15 min': 1.15, 'Last 30 min': 1.08, 'Last 1 hour': 0, 'Today': 0,
  };
  const timeThreshold = TIME_ACTIVITY_THRESHOLD[filters.time] ?? 0;

  const filteredJunctions = junctions.filter(j => {
    if (timeThreshold > 0 && j.congestionFactor < timeThreshold) return false;
    if (filters.risk !== 'All' && j.riskLevel !== filters.risk.toUpperCase()) return false;
    if (filters.traffic !== 'All') {
      const band = CONGESTION_BANDS[filters.traffic];
      if (band && !band(j.congestionFactor)) return false;
    }
    if (filters.area !== 'Nagpur City') {
      const kw = AREA_KEYWORDS[filters.area];
      if (kw && !j.name.includes(kw) && !j.shortName.includes(kw)) return false;
    }
    return true;
  });

  const filteredOfficers = officers.filter(o => {
    if (filters.officers !== 'All') {
      if (o.status !== OFFICER_STATUS_MAP[filters.officers]) return false;
    }
    if (filters.area !== 'Nagpur City') {
      const kw = AREA_KEYWORDS[filters.area];
      if (kw && o.locationName && !o.locationName.includes(kw)) return false;
    }
    return true;
  });

  const filteredAllocations = allocations.filter(a =>
    filteredJunctions.some(j => j.id === a.junctionId)
  );
  const filteredRoutes = routes.filter(r =>
    filteredJunctions.some(j => j.id === r.junctionId)
  );

  // derived
  const sortedJunctions = [...filteredJunctions].sort((a, b) => b.riskScore - a.riskScore);
  const allocation: OfficerAllocation | null =
    allocations.find(a => a.junctionId === selectedJunction?.id) ?? null;
  const uncoveredJunctions = coverage
    ? filteredJunctions.filter(j => coverage.uncoveredJunctions.includes(j.id))
    : [];

  // time ticker
  useEffect(() => {
    const id = setInterval(() => {
      const now = new Date();
      setLastUpdated(now.toLocaleTimeString('en-US', { hour12: true }));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const handleSelectJunction = useCallback((j: Junction) => {
    setSelectedJunction(j);
    setSelectedOfficer(null);
    setActiveRoute(null);
  }, []);

  const handleSelectOfficer = useCallback((o: Officer) => {
    setSelectedOfficer(o);
  }, []);

  const handleViewRoute = useCallback((j: Junction) => {
    const officer = officers.find(o => o.id === j.assignedOfficer);
    if (!officer) return;
    const route = routes.find(r => r.officerId === officer.id && r.junctionId === j.id) ?? null;
    setActiveRoute(route);
    setSelectedJunction(j);
  }, [officers, routes]);

  const handleRefresh = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    try {
      const fresh = await refreshAllData(demoMode);
      setJunctionsLocal(fresh.junctions);
      setOfficersLocal(fresh.officers);
      setJunctions(fresh.junctions);
      setOfficers(fresh.officers);
      setRoutesHook(fresh.routes);
      setTrafficHook(fresh.traffic);
      setCoverageHook(fresh.coverage);
      setBaselineHook(fresh.baseline);
      setAllocationsHook(fresh.allocations);
      setLastUpdated(new Date().toLocaleTimeString('en-US', { hour12: true }));
      setRefreshToast('success');
      setTimeout(() => setRefreshToast(null), 2500);
    } catch {
      setRefreshToast('error');
      setTimeout(() => setRefreshToast(null), 3000);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleLockOfficer = async (officerId: string) => {
    await lockOfficer(officerId, demoMode);
    updateOfficers(prev => prev.map(o =>
      o.id === officerId ? { ...o, locked: true, status: 'LOCKED' as const } : o
    ));
  };

  const handleUnlockOfficer = async (officerId: string) => {
    await unlockOfficer(officerId, demoMode);
    updateOfficers(prev => prev.map(o =>
      o.id === officerId ? { ...o, locked: false, status: 'ASSIGNED' as const } : o
    ));
  };

  const handleSimulate = async () => {
    setIncidentStep('processing');
    setProcessStep(0);
    setIncidentJunctionId(incidentForm.junction);

    const now = new Date();
    const baseTime = now.getTime();
    const newTimeline = PROCESS_STEPS.map((label, i) => ({
      time: new Date(baseTime + i * 1000).toLocaleTimeString('en-US', { hour12: false }),
      label: i === 0 ? 'Incident detected' : label,
    }));

    // Animate steps
    for (let i = 0; i <= PROCESS_STEPS.length; i++) {
      await new Promise(r => setTimeout(r, 700));
      setProcessStep(i);
      if (i < newTimeline.length) {
        setTimeline(prev => [...prev, newTimeline[i]]);
      }
    }

    const result = await simulateIncident({
      junctionId: incidentForm.junction,
      incidentType: incidentForm.type,
      severity: incidentForm.severity,
      congestionImpact: incidentForm.congestion,
    }, demoMode);

    setIncidentResult(result);

    // Update junction risk/congestion from DeploymentResult
    updateJunctions(prev => prev.map(j =>
      j.id === result.affectedJunction
        ? { ...j, riskScore: result.newRiskScore, riskLevel: 'CRITICAL' as const, congestionFactor: result.newCongestion, assignedOfficer: result.newOfficers[0]?.officerId ?? j.assignedOfficer }
        : j
    ));

    // Update officer assignments from DeploymentResult
    if (result.newOfficers.length > 0) {
      updateOfficers(prev => prev.map(o => {
        const change = result.newOfficers.find(c => c.officerId === o.id);
        if (!change) return o;
        return { ...o, assignedJunction: change.toJunction, status: 'ASSIGNED' as const };
      }));
    }

    setIncidentStep('result');
  };

  const handleCloseIncident = () => {
    setShowIncidentModal(false);
    setIncidentStep('form');
    setIncidentResult(null);
    setProcessStep(0);
  };

  const activeCount = filteredOfficers.filter(o => o.status !== 'OFFLINE').length;
  const highRiskCount = filteredJunctions.filter(j => j.riskLevel === 'CRITICAL' || j.riskLevel === 'HIGH').length;

  // ─── render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-primary)', overflow: 'hidden' }}>
      {/* HEADER */}
      <Header
        lastUpdated={dataLastUpdated ? dataLastUpdated.toLocaleTimeString('en-US', { hour12: true }) : lastUpdated}
        isRefreshing={isRefreshing || dataStatus === 'loading'}
        isOnline={isBackendOnline}
        connectionMode={connectionMode}
        onRefresh={handleRefresh}
        demoMode={demoMode}
        onToggleDemo={() => setDemoMode(d => !d)}
        onToggleSidebar={() => setSidebarOpen(s => !s)}
      />

      {/* Backend error banner — non-intrusive, dismissible */}
      {dataStatus === 'error' && dataError && (
        <div style={{
          background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.25)',
          padding: '5px 16px', display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0,
        }}>
          <AlertTriangle size={11} style={{ color: '#fca5a5', flexShrink: 0 }} />
          <span style={{ fontSize: 11, color: '#fca5a5' }}>
            ⚠ Unable to connect to NovaRoute backend — showing last known data.
          </span>
          <button
            onClick={handleRefresh}
            style={{
              marginLeft: 'auto', background: 'transparent', border: '1px solid rgba(239,68,68,0.3)',
              color: '#fca5a5', borderRadius: 3, padding: '2px 8px', fontSize: 10, cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Refresh toast */}
      {refreshToast && (
        <div style={{
          position: 'fixed', bottom: 20, right: 20, zIndex: 10000,
          background: refreshToast === 'success' ? 'rgba(0,208,132,0.12)' : 'rgba(239,68,68,0.1)',
          border: `1px solid ${refreshToast === 'success' ? 'rgba(0,208,132,0.3)' : 'rgba(239,68,68,0.25)'}`,
          color: refreshToast === 'success' ? '#00d084' : '#fca5a5',
          borderRadius: 4, padding: '7px 14px', fontSize: 11, fontWeight: 500,
          display: 'flex', alignItems: 'center', gap: 6,
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
          animation: 'fade-in 0.2s ease-out',
        }}>
          {refreshToast === 'success'
            ? <><CheckCircle size={12} /> Data refreshed</>
            : <><AlertCircle size={12} /> Refresh failed — showing cached data</>}
        </div>
      )}

      {/* BODY */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* SIDEBAR */}
        {sidebarOpen && (
          <Sidebar
            active={activeSidebar}
            onSelect={setActiveSidebar}
          />
        )}

        {/* MAIN CONTENT — view router */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

          {activeSidebar === 'dashboard' && (
            <>
              <FilterBar
                filters={filters}
                onSetFilter={setFilter}
                onReset={resetFilters}
                layers={layers}
                onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
                onSimulate={() => { setShowIncidentModal(true); setIncidentStep('form'); }}
              />
              {coverage && traffic ? (
                <KPIRow highRiskCount={highRiskCount} coverage={coverage} activeCount={activeCount} totalOfficers={filteredOfficers.length} traffic={traffic} />
              ) : (
                <KPILoading />
              )}
              <div style={{ flex: 1, display: 'flex', overflow: 'hidden', gap: 1 }}>
                <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
                  <MapView junctions={filteredJunctions} officers={filteredOfficers} routes={filteredRoutes} layers={layers}
                    selectedJunction={selectedJunction} activeRoute={activeRoute} incidentJunctionId={incidentJunctionId}
                    onSelectJunction={handleSelectJunction} onSelectOfficer={handleSelectOfficer} onViewRoute={handleViewRoute} />
                  <MapOverlayControls layers={layers} onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))} />
                </div>
                <div style={{ width: showExplainability ? 0 : 320, minWidth: showExplainability ? 0 : 320, overflowY: 'auto', overflowX: 'hidden', background: 'var(--bg-panel)', borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', transition: 'width 0.25s ease' }}>
                  <RiskRankingPanel junctions={sortedJunctions} selectedJunction={selectedJunction} onSelect={handleSelectJunction} />
                  <Divider />
                  <OfficerPanel officers={filteredOfficers} selectedOfficer={selectedOfficer} onSelect={handleSelectOfficer} onLock={handleLockOfficer} onUnlock={handleUnlockOfficer} />
                  <Divider />
                  {selectedJunction && allocation && (<><AllocationPanel junction={selectedJunction} allocation={allocation} officers={officers} onExplain={() => setShowExplainability(true)} onViewRoute={() => handleViewRoute(selectedJunction)} /><Divider /></>)}
                  {coverage ? <CoveragePanel coverage={coverage} uncoveredJunctions={uncoveredJunctions} onSelectJunction={handleSelectJunction} /> : <PanelLoading label="Coverage" />}
                  <Divider />
                  {baseline ? <BaselinePanel baseline={baseline} /> : <PanelLoading label="Performance" />}
                  <Divider />
                  {traffic ? <TrafficPanel traffic={traffic} /> : <PanelLoading label="Live Traffic" />}
                  <Divider />
                  <CameraFeedPanel />
                </div>
                {showExplainability && selectedJunction && allocation && (
                  <ExplainabilityPanel junction={selectedJunction} allocation={allocation} officers={officers} onClose={() => setShowExplainability(false)} onViewRoute={() => handleViewRoute(selectedJunction)} />
                )}
              </div>
            </>
          )}

          {activeSidebar === 'risk' && (
            <RiskIntelligenceView
              junctions={sortedJunctions}
              selectedJunction={selectedJunction}
              layers={layers}
              onSelectJunction={handleSelectJunction}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'traffic' && (
            <LiveTrafficView
              traffic={traffic}
              junctions={filteredJunctions}
              layers={layers}
              onSelectJunction={handleSelectJunction}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'officers' && (
            <OfficersView
              officers={filteredOfficers}
              junctions={filteredJunctions}
              selectedOfficer={selectedOfficer}
              layers={layers}
              onSelectOfficer={handleSelectOfficer}
              onLock={handleLockOfficer}
              onUnlock={handleUnlockOfficer}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'deployment' && (
            <DeploymentView
              allocations={filteredAllocations}
              officers={officers}
              junctions={filteredJunctions}
              routes={filteredRoutes}
              activeRoute={activeRoute}
              selectedJunction={selectedJunction}
              layers={layers}
              onSelectJunction={handleSelectJunction}
              onViewRoute={handleViewRoute}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'incidents' && (
            <IncidentsView
              junctions={filteredJunctions}
              incidentResult={incidentResult}
              incidentJunctionId={incidentJunctionId}
              layers={layers}
              onSimulate={() => { setShowIncidentModal(true); setIncidentStep('form'); }}
              onSelectJunction={handleSelectJunction}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'coverage' && (
            <CoverageView
              coverage={coverage}
              junctions={filteredJunctions}
              officers={filteredOfficers}
              allocations={filteredAllocations}
              uncoveredJunctions={uncoveredJunctions}
              layers={layers}
              onSelectJunction={handleSelectJunction}
              onToggleLayer={(k) => setLayers(prev => ({ ...prev, [k]: !prev[k as keyof typeof prev] }))}
            />
          )}

          {activeSidebar === 'performance' && (
            <PerformanceView
              baseline={baseline}
              coverage={coverage}
              traffic={traffic}
              officers={filteredOfficers}
              junctions={filteredJunctions}
              allocations={filteredAllocations}
            />
          )}

        </div>
      </div>

      {/* INCIDENT MODAL */}
      {showIncidentModal && (
        <IncidentModal
          step={incidentStep}
          form={incidentForm}
          junctions={junctions}
          processStep={processStep}
          result={incidentResult}
          timeline={timeline}
          onFormChange={(k, v) => setIncidentForm(f => ({ ...f, [k]: v }))}
          onSimulate={handleSimulate}
          onClose={handleCloseIncident}
        />
      )}

      {/* OFFICER DETAIL MODAL */}
      {selectedOfficer && (
        <OfficerDetailModal
          officer={selectedOfficer}
          junction={junctions.find(j => j.id === selectedOfficer.assignedJunction) ?? null}
          onClose={() => setSelectedOfficer(null)}
          onLock={handleLockOfficer}
          onUnlock={handleUnlockOfficer}
          onViewRoute={() => {
            if (selectedOfficer.assignedJunction) {
              const j = junctions.find(x => x.id === selectedOfficer.assignedJunction);
              if (j) handleViewRoute(j);
            }
            setSelectedOfficer(null);
          }}
        />
      )}
    </div>
  );
}

// ─── HEADER ──────────────────────────────────────────────────────────────────
function Header({ lastUpdated, isRefreshing, isOnline, connectionMode, onRefresh, demoMode, onToggleDemo, onToggleSidebar }: {
  lastUpdated: string; isRefreshing: boolean; isOnline: boolean;
  connectionMode: import('./hooks/useNovaRouteData').ConnectionMode;
  onRefresh: () => void; demoMode: boolean;
  onToggleDemo: () => void; onToggleSidebar: () => void;
}) {
  const statusLabel = connectionMode === 'live' ? 'BACKEND CONNECTED' : connectionMode === 'fallback' ? 'DEMO FALLBACK' : 'DEMO MODE';
  const statusColor = connectionMode === 'live' ? '#00d084' : connectionMode === 'fallback' ? '#f59e0b' : '#3b82f6';
  return (
    <header style={{
      height: 52, background: '#080e1e', borderBottom: '1px solid #1c3050',
      display: 'flex', alignItems: 'center', padding: '0 16px', gap: 12, flexShrink: 0,
    }}>
      <button onClick={onToggleSidebar} style={{ background: 'none', border: 'none', color: '#4a6080', cursor: 'pointer', padding: 4 }}>
        <Menu size={16} />
      </button>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: '#e2e8f0', letterSpacing: '0.05em' }}>NOVAROUTE<span style={{ color: '#3b82f6' }}>.AI</span></span>
        <span style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.08em' }}>AI TRAFFIC RISK & POLICE DEPLOYMENT</span>
        <span style={{ fontSize: 10, color: '#2a4060', letterSpacing: '0.08em' }}>/ NAGPUR CITY</span>
      </div>

      <div style={{ marginLeft: 16, display: 'flex', alignItems: 'center', gap: 5 }}>
        <PulseDot color={statusColor} />
        <span style={{ fontSize: 10, color: statusColor }}>{statusLabel}</span>
        <span style={{ fontSize: 10, color: '#2a4060', marginLeft: 4 }}>Last updated: {lastUpdated}</span>
      </div>

      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* Demo mode toggle */}
        <button
          onClick={onToggleDemo}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            background: demoMode ? 'rgba(251,191,36,0.1)' : '#0c1428',
            border: `1px solid ${demoMode ? '#f59e0b40' : '#1c3050'}`,
            color: demoMode ? '#f59e0b' : '#4a6080',
            borderRadius: 3, padding: '3px 8px', fontSize: 10, fontWeight: 600,
            cursor: 'pointer', letterSpacing: '0.05em',
          }}
        >
          <Zap size={10} />
          DEMO {demoMode ? 'ON' : 'OFF'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <PulseDot color="#00d084" />
          <span style={{ fontSize: 11, color: '#4a6080' }}>Traffic Feed</span>
          <span style={{ fontSize: 10, color: '#00d084', fontWeight: 600 }}>● LIVE</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ fontSize: 11, color: '#4a6080' }}>Network</span>
          <span style={{ fontSize: 10, color: '#00d084', fontWeight: 600 }}>● CONNECTED</span>
        </div>

        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            background: '#0f1e38', border: '1px solid #1c3050',
            color: '#7a9abf', borderRadius: 3, padding: '4px 10px',
            fontSize: 11, cursor: 'pointer',
          }}
        >
          <RefreshCw size={11} style={{ animation: isRefreshing ? 'spin 1s linear infinite' : undefined }} />
          Refresh
        </button>
      </div>
    </header>
  );
}

// ─── SIDEBAR ─────────────────────────────────────────────────────────────────
function Sidebar({ active, onSelect }: { active: string; onSelect: (id: string) => void }) {
  return (
    <aside style={{
      width: 200, background: '#080e1e', borderRight: '1px solid #1c3050',
      display: 'flex', flexDirection: 'column', flexShrink: 0,
    }}>
      <div style={{ padding: '12px 16px 8px' }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#3b82f6', letterSpacing: '0.08em' }}>NovaRoute<span style={{ color: '#4a6080' }}>.AI</span></div>
      </div>
      <nav style={{ flex: 1, padding: '4px 8px' }}>
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => onSelect(id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 9,
              width: '100%', padding: '7px 10px', borderRadius: 3, marginBottom: 1,
              background: active === id ? '#111d35' : 'transparent',
              border: active === id ? '1px solid #1c3050' : '1px solid transparent',
              color: active === id ? '#c8d8f0' : '#4a6080',
              fontSize: 12, cursor: 'pointer', textAlign: 'left',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </nav>
      <div style={{ padding: '12px 16px', borderTop: '1px solid #1c3050' }}>
        <div style={{ fontSize: 10, color: '#2a4060', marginBottom: 4, letterSpacing: '0.08em' }}>SYSTEM STATUS</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <PulseDot color="#00d084" />
          <span style={{ fontSize: 10, color: '#4a6080' }}>All systems operational</span>
        </div>
      </div>
    </aside>
  );
}

// ─── FILTER BAR ───────────────────────────────────────────────────────────────
const FILTER_OPTIONS: Record<string, string[]> = {
  time:     ['Live', 'Last 15 min', 'Last 30 min', 'Last 1 hour', 'Today'],
  risk:     ['All', 'Critical', 'High', 'Medium', 'Low'],
  traffic:  ['All', 'Low', 'Moderate', 'High', 'Severe'],
  officers: ['All', 'Available', 'Deployed', 'Locked', 'Offline'],
  area:     ['Nagpur City', 'Wardha Rd', 'Itwari', 'Cotton Market', 'Sitabuldi', 'Sadar', 'Mankapur', 'Koradi Rd'],
};
const FILTER_LABELS: Record<string, string> = {
  time: 'Time', risk: 'Risk', traffic: 'Traffic', officers: 'Officers', area: 'Area',
};
const FILTER_KEYS = ['time', 'risk', 'traffic', 'officers', 'area'] as const;

function FilterBar({ filters, onSetFilter, onReset, layers, onToggleLayer, onSimulate }: {
  filters: FilterState;
  onSetFilter: (key: keyof FilterState, value: string) => void;
  onReset: () => void;
  layers: LayerState;
  onToggleLayer: (k: string) => void;
  onSimulate: () => void;
}) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside the filter bar
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenKey(null);
      }
    }
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  const isActive = (key: string) => filters[key as keyof FilterState] !== DEFAULT_FILTERS[key as keyof FilterState];

  return (
    <div
      ref={barRef}
      style={{
        height: 40, background: '#0a1020', borderBottom: '1px solid #1c3050',
        display: 'flex', alignItems: 'center', padding: '0 12px', gap: 8,
        flexShrink: 0, position: 'relative', zIndex: 200,
      }}
    >
      <span style={{ fontSize: 10, color: '#2a4060', marginRight: 4, letterSpacing: '0.08em' }}>FILTERS</span>

      {FILTER_KEYS.map(key => {
        const value = filters[key];
        const active = isActive(key);
        const isOpen = openKey === key;
        const opts = FILTER_OPTIONS[key];
        return (
          <div key={key} style={{ position: 'relative' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: 10, color: active ? '#7a9abf' : '#4a6080' }}>{FILTER_LABELS[key]}</span>
              <button
                onClick={() => setOpenKey(isOpen ? null : key)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  background: active ? '#0f1e38' : '#0f1629',
                  border: `1px solid ${active ? '#2563eb50' : '#1c3050'}`,
                  color: active ? '#93c5fd' : '#7a9abf',
                  borderRadius: 3, padding: '2px 7px', fontSize: 10, cursor: 'pointer',
                }}
              >
                {value} <ChevronDown size={9} style={{ transition: 'transform 0.15s', transform: isOpen ? 'rotate(180deg)' : undefined }} />
              </button>
            </div>

            {isOpen && (
              <div style={{
                position: 'absolute', top: '100%', left: 0, marginTop: 4,
                background: '#0c1428', border: '1px solid #1c3050', borderRadius: 4,
                boxShadow: '0 8px 24px rgba(0,0,0,0.8)', zIndex: 9999, minWidth: 130, overflow: 'hidden',
              }}>
                {opts.map(opt => (
                  <button
                    key={opt}
                    onClick={() => { onSetFilter(key, opt); setOpenKey(null); }}
                    style={{
                      display: 'block', width: '100%', padding: '6px 10px',
                      background: opt === value ? '#111d35' : 'transparent',
                      border: 'none', borderBottom: '1px solid #0f1629',
                      color: opt === value ? '#93c5fd' : '#7a9abf',
                      fontSize: 11, cursor: 'pointer', textAlign: 'left',
                      fontFamily: 'Inter, sans-serif',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = '#111d35'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = opt === value ? '#111d35' : 'transparent'; }}
                  >
                    {opt === value && <span style={{ color: '#3b82f6', marginRight: 5 }}>●</span>}
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Reset — only show when any filter is non-default */}
      {FILTER_KEYS.some(k => isActive(k)) && (
        <button
          onClick={onReset}
          style={{
            display: 'flex', alignItems: 'center', gap: 3,
            background: 'transparent', border: '1px solid #2a4060',
            color: '#4a6080', borderRadius: 3, padding: '2px 7px',
            fontSize: 10, cursor: 'pointer',
          }}
        >
          <X size={8} /> Reset
        </button>
      )}

      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
        {Object.entries(layers).map(([key, on]) => (
          <button
            key={key}
            onClick={() => onToggleLayer(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              background: on ? '#111d35' : 'transparent',
              border: `1px solid ${on ? '#1e3050' : '#1c2030'}`,
              color: on ? '#7a9abf' : '#2a4060',
              borderRadius: 3, padding: '2px 7px', fontSize: 10, cursor: 'pointer',
            }}
          >
            {on ? <Eye size={9} /> : <EyeOff size={9} />}
            {key.charAt(0).toUpperCase() + key.slice(1)}
          </button>
        ))}

        <button
          onClick={onSimulate}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.35)',
            color: '#fca5a5', borderRadius: 3, padding: '3px 10px',
            fontSize: 10, fontWeight: 600, cursor: 'pointer', letterSpacing: '0.04em',
          }}
        >
          <AlertCircle size={11} />
          SIMULATE INCIDENT
        </button>
      </div>
    </div>
  );
}

// ─── KPI CARDS ────────────────────────────────────────────────────────────────
function KPIRow({ highRiskCount, coverage, activeCount, totalOfficers, traffic }: {
  highRiskCount: number; coverage: any; activeCount: number; totalOfficers: number; traffic: any;
}) {
  return (
    <div style={{
      display: 'flex', gap: 1, height: 76, flexShrink: 0,
      background: '#080e1e', borderBottom: '1px solid #1c3050', padding: '0 1px',
    }}>
      <KPICard
        label="HIGH-RISK JUNCTIONS"
        value={highRiskCount.toString()}
        sub="↑ 12% vs previous period"
        subColor="#ff8c42"
        accentColor="#ff3a3a"
        icon={<AlertTriangle size={14} />}
      />
      <KPICard
        label="COVERAGE"
        value={`${coverage.coveragePct}%`}
        sub={`Within ${coverage.thresholdMinutes} min`}
        subColor="#00d084"
        accentColor="#00c896"
        icon={<Shield size={14} />}
      />
      <KPICard
        label="AVG RESPONSE"
        value={`${coverage.avgResponse} min`}
        sub="↓ 18% improved"
        subColor="#00d084"
        accentColor="#3b82f6"
        icon={<Activity size={14} />}
      />
      <KPICard
        label="OFFICERS ACTIVE"
        value={`${activeCount} / ${totalOfficers}`}
        sub={`● ${activeCount} available`}
        subColor="#00d084"
        accentColor="#00d084"
        icon={<Users size={14} />}
      />
      <KPICard
        label="NETWORK CONGESTION"
        value={traffic.congestionLevel}
        sub={`${traffic.congestionFactor}×`}
        subColor="#ffd700"
        accentColor="#ffd700"
        icon={<Radio size={14} />}
      />
    </div>
  );
}

function KPICard({ label, value, sub, subColor, accentColor, icon }: {
  label: string; value: string; sub: string; subColor: string; accentColor: string; icon: React.ReactNode;
}) {
  return (
    <div style={{
      flex: 1, padding: '10px 14px',
      background: '#0c1428', borderRight: '1px solid #1c3050',
      display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span style={{ fontSize: 9, color: '#4a6080', letterSpacing: '0.1em', fontWeight: 600 }}>{label}</span>
        <span style={{ color: accentColor, opacity: 0.7 }}>{icon}</span>
      </div>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, fontWeight: 700, color: accentColor, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: subColor }}>{sub}</div>
    </div>
  );
}

// ─── MAP OVERLAY CONTROLS ────────────────────────────────────────────────────
function MapOverlayControls({ layers, onToggleLayer }: {
  layers: LayerState; onToggleLayer: (k: string) => void;
}) {
  return (
    <div style={{
      position: 'absolute', top: 10, right: 10, zIndex: 1000,
      background: 'rgba(8,14,30,0.92)', border: '1px solid #1c3050',
      borderRadius: 4, padding: '8px 10px', minWidth: 140,
    }}>
      <div style={{ fontSize: 9, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 6, fontWeight: 600 }}>MAP LAYERS</div>
      {Object.entries(layers).map(([key, on]) => (
        <label key={key} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', marginBottom: 3 }}>
          <input
            type="checkbox" checked={on}
            onChange={() => onToggleLayer(key)}
            style={{ accentColor: '#3b82f6', width: 11, height: 11 }}
          />
          <span style={{ fontSize: 11, color: on ? '#c8d8f0' : '#4a6080' }}>
            {key.charAt(0).toUpperCase() + key.slice(1)}
          </span>
        </label>
      ))}
    </div>
  );
}

// ─── RISK RANKING PANEL ───────────────────────────────────────────────────────
function RiskRankingPanel({ junctions, selectedJunction, onSelect }: {
  junctions: Junction[]; selectedJunction: Junction | null; onSelect: (j: Junction) => void;
}) {
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>TOP RISK LOCATIONS</SectionTitle>
      {junctions.slice(0, 8).map((j, i) => {
        const isSelected = selectedJunction?.id === j.id;
        return (
          <button
            key={j.id}
            onClick={() => onSelect(j)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '6px 8px', marginBottom: 2, borderRadius: 3,
              background: isSelected ? '#111d35' : 'transparent',
              border: `1px solid ${isSelected ? '#1e3050' : 'transparent'}`,
              cursor: 'pointer', textAlign: 'left',
            }}
          >
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#2a4060', width: 16, flexShrink: 0 }}>
              {String(i + 1).padStart(2, '0')}
            </span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', fontWeight: 600 }}>{j.id}</div>
              <div style={{ fontSize: 10, color: '#4a6080', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{j.shortName}</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2, flexShrink: 0 }}>
              <RiskBadge level={j.riskLevel} />
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: RISK_COLOR[j.riskLevel], fontWeight: 700 }}>{j.riskScore}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ─── OFFICER PANEL ────────────────────────────────────────────────────────────
function OfficerPanel({ officers, selectedOfficer, onSelect, onLock, onUnlock }: {
  officers: Officer[]; selectedOfficer: Officer | null;
  onSelect: (o: Officer) => void; onLock: (id: string) => void; onUnlock: (id: string) => void;
}) {
  const active = officers.filter(o => o.status !== 'OFFLINE');
  const total = officers.length;
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <SectionTitle>ACTIVE OFFICERS</SectionTitle>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#00d084' }}>{active.length} / {total}</span>
      </div>
      {officers.slice(0, 8).map(o => {
        const isSelected = selectedOfficer?.id === o.id;
        return (
          <div
            key={o.id}
            onClick={() => onSelect(o)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', marginBottom: 2,
              background: isSelected ? '#111d35' : 'transparent',
              border: `1px solid ${isSelected ? '#1e3050' : 'transparent'}`,
              borderRadius: 3, cursor: 'pointer',
            }}
          >
            <PulseDot color={STATUS_COLOR[o.status]} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', fontWeight: 600 }}>{o.badge}</span>
                {o.locked && <Lock size={9} style={{ color: '#6b7280' }} />}
              </div>
              <div style={{ fontSize: 10, color: '#4a6080' }}>{o.assignedJunction ?? '—'} · {o.responseTime != null ? `${o.responseTime} min` : 'offline'}</div>
            </div>
            <span style={{
              fontSize: 9, color: STATUS_COLOR[o.status],
              background: `${STATUS_COLOR[o.status]}18`, border: `1px solid ${STATUS_COLOR[o.status]}30`,
              borderRadius: 2, padding: '1px 5px', fontWeight: 600, letterSpacing: '0.04em',
            }}>{o.status}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── ALLOCATION PANEL ─────────────────────────────────────────────────────────
function AllocationPanel({ junction, allocation, officers, onExplain, onViewRoute }: {
  junction: Junction; allocation: any; officers: Officer[];
  onExplain: () => void; onViewRoute: () => void;
}) {
  const assignedOfficer = officers.find(o => o.id === allocation.officerId);
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>CURRENT DEPLOYMENT</SectionTitle>
      <div style={{ background: '#111d35', border: '1px solid #1c3050', borderRadius: 4, padding: '10px 12px', marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#c8d8f0', fontWeight: 700 }}>{junction.id}</span>
          <RiskBadge level={junction.riskLevel} />
        </div>
        <div style={{ fontSize: 10, color: '#4a6080', marginBottom: 8 }}>{junction.name}</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 10, color: '#4a6080' }}>Assigned</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#00d084', fontWeight: 600 }}>{assignedOfficer?.badge}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={{ fontSize: 10, color: '#4a6080' }}>Response time</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#3b82f6' }}>{allocation.responseTime} min</span>
        </div>

        <div style={{ fontSize: 10, color: '#2a4060', marginBottom: 5 }}>Alternative officers:</div>
        {allocation.alternatives.map((alt: any) => {
          const o = officers.find(x => x.id === alt.officerId);
          return (
            <div key={alt.officerId} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
              <span style={{ fontSize: 10, color: '#4a6080' }}>{o?.badge ?? alt.officerId}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#2a4060' }}>{alt.responseTime} min</span>
            </div>
          );
        })}

        <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
          <button onClick={onExplain} style={{
            flex: 1, padding: '5px 0', background: '#1e3a6e', border: '1px solid #2563eb',
            color: '#93c5fd', borderRadius: 3, fontSize: 10, fontWeight: 500, cursor: 'pointer',
          }}>
            WHY THIS?
          </button>
          <button onClick={onViewRoute} style={{
            flex: 1, padding: '5px 0', background: '#0f2a1a', border: '1px solid #166534',
            color: '#4ade80', borderRadius: 3, fontSize: 10, fontWeight: 500, cursor: 'pointer',
          }}>
            VIEW ROUTE
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── COVERAGE PANEL ───────────────────────────────────────────────────────────
function CoveragePanel({ coverage, uncoveredJunctions, onSelectJunction }: {
  coverage: any; uncoveredJunctions: Junction[]; onSelectJunction: (j: Junction) => void;
}) {
  const pct = coverage.coveragePct;
  const circumference = 2 * Math.PI * 28;
  const offset = circumference * (1 - pct / 100);
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>DEPLOYMENT COVERAGE</SectionTitle>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 10 }}>
        <svg width={70} height={70} style={{ flexShrink: 0 }}>
          <circle cx={35} cy={35} r={28} fill="none" stroke="#1c3050" strokeWidth={5} />
          <circle
            cx={35} cy={35} r={28} fill="none" stroke="#00c896" strokeWidth={5}
            strokeDasharray={circumference} strokeDashoffset={offset}
            strokeLinecap="round" transform="rotate(-90 35 35)"
          />
          <text x={35} y={39} textAnchor="middle" style={{ fill: '#c8d8f0', fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700 }}>{pct}%</text>
        </svg>
        <div>
          <div style={{ fontSize: 11, color: '#c8d8f0', marginBottom: 4 }}>
            {coverage.coveredJunctions} / {coverage.totalJunctions} junctions
          </div>
          <div style={{ fontSize: 10, color: '#4a6080' }}>within {coverage.thresholdMinutes} min</div>
          <div style={{ marginTop: 6, display: 'flex', gap: 12 }}>
            {[
              { label: 'Avg', value: `${coverage.avgResponse}m` },
              { label: 'Worst', value: `${coverage.worstResponse}m` },
              { label: 'Uncov.', value: coverage.uncoveredCount },
            ].map(({ label, value }) => (
              <div key={label}>
                <div style={{ fontSize: 9, color: '#2a4060' }}>{label}</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#c8d8f0' }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {uncoveredJunctions.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: '#f59e0b', marginBottom: 5, display: 'flex', alignItems: 'center', gap: 4 }}>
            <AlertTriangle size={10} />
            {uncoveredJunctions.length} locations exceed {coverage.thresholdMinutes}-min threshold
          </div>
          {uncoveredJunctions.map(j => (
            <button
              key={j.id}
              onClick={() => onSelectJunction(j)}
              style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                width: '100%', padding: '5px 8px', marginBottom: 2,
                background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)',
                borderRadius: 3, cursor: 'pointer',
              }}
            >
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#f59e0b' }}>{j.id}</span>
              <span style={{ fontSize: 10, color: '#4a6080' }}>{j.shortName}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#ff3a3a' }}>
                {'>6m'}
              </span>
            </button>
          ))}
        </>
      )}
    </div>
  );
}

// ─── BASELINE PANEL ───────────────────────────────────────────────────────────
function BaselinePanel({ baseline }: { baseline: any }) {
  const metrics = [
    { label: 'Avg Response', stat: 'avgResponse', unit: 'min', better: 'low' },
    { label: 'Coverage <6m', stat: 'coveragePct', unit: '%', better: 'high' },
    { label: 'Covered', stat: 'coveredCount', unit: `/${baseline.static.total}`, better: 'high' },
  ];
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>PERFORMANCE COMPARISON</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px 80px', gap: '4px 0', marginBottom: 6 }}>
        <div />
        <div style={{ fontSize: 9, color: '#4a6080', textAlign: 'center', letterSpacing: '0.06em' }}>STATIC</div>
        <div style={{ fontSize: 9, color: '#3b82f6', textAlign: 'center', letterSpacing: '0.06em' }}>NOVAROUTE</div>
        {metrics.map(m => {
          const sv = baseline.static[m.stat];
          const nv = baseline.novaroute[m.stat];
          const novaBetter = m.better === 'low' ? nv < sv : nv > sv;
          return [
            <div key={`${m.stat}-l`} style={{ fontSize: 10, color: '#4a6080', padding: '3px 0' }}>{m.label}</div>,
            <div key={`${m.stat}-s`} style={{ textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#4a6080', padding: '3px 0' }}>{sv}{m.unit}</div>,
            <div key={`${m.stat}-n`} style={{
              textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
              color: novaBetter ? '#00d084' : '#ff3a3a', fontWeight: 600, padding: '3px 0',
            }}>{nv}{m.unit}</div>,
          ];
        }).flat()}
      </div>
      <div style={{ fontSize: 9, color: '#2a4060', display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <span>Baseline: Static police posting</span>
        <span style={{ color: '#1e40af' }}>NovaRoute: Dynamic AI allocation</span>
      </div>
      <div style={{ marginTop: 8 }}>
        {['Avg Response', 'Coverage'].map((label, i) => {
          const svPct = i === 0
            ? (baseline.novaroute.avgResponse / baseline.static.avgResponse) * 100
            : (baseline.static.coveragePct / 100) * 100;
          const nvPct = i === 0 ? 100 : (baseline.novaroute.coveragePct / 100) * 100;
          return (
            <div key={label} style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 10, color: '#4a6080', marginBottom: 3 }}>{label}</div>
              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <div style={{ flex: 1, height: 8, background: '#1c3050', borderRadius: 2, overflow: 'hidden', position: 'relative' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${svPct}%`, background: '#374151', borderRadius: 2 }} />
                </div>
                <div style={{ flex: 1, height: 8, background: '#1c3050', borderRadius: 2, overflow: 'hidden', position: 'relative' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${nvPct}%`, background: '#3b82f6', borderRadius: 2 }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── TRAFFIC PANEL ────────────────────────────────────────────────────────────
function TrafficPanel({ traffic }: { traffic: any }) {
  const trendMax = Math.max(...traffic.trend);
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>LIVE TRAFFIC</SectionTitle>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <PulseDot color="#00d084" />
          <span style={{ fontSize: 10, color: '#4a6080' }}>Network</span>
          <span style={{ fontSize: 10, color: '#00d084', fontWeight: 600 }}>LIVE</span>
        </div>
        <span style={{ fontSize: 10, color: '#2a4060' }}>{traffic.lastUpdated}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
        {[
          { label: 'Vehicle Count', value: traffic.vehicleCount.toLocaleString() },
          { label: 'Avg Density', value: traffic.avgDensity.toFixed(2) },
          { label: 'Congestion', value: traffic.congestionLevel },
          { label: 'Factor', value: `${traffic.congestionFactor}×` },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: '#111d35', border: '1px solid #1c3050', borderRadius: 3, padding: '6px 8px' }}>
            <div style={{ fontSize: 9, color: '#4a6080', marginBottom: 2 }}>{label}</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#c8d8f0', fontWeight: 600 }}>{value}</div>
          </div>
        ))}
      </div>
      <div style={{ height: 36, display: 'flex', alignItems: 'flex-end', gap: 2 }}>
        {traffic.trend.map((v: number, i: number) => (
          <div
            key={i}
            style={{
              flex: 1, background: '#2563eb',
              height: `${(v / trendMax) * 100}%`,
              borderRadius: '2px 2px 0 0', opacity: 0.5 + (i / traffic.trend.length) * 0.5,
              minHeight: 2,
            }}
          />
        ))}
      </div>
    </div>
  );
}

// ─── CAMERA FEED PANEL ────────────────────────────────────────────────────────
function CameraFeedPanel() {
  const [camera] = useState({ id: 'CAM-07', location: 'Wardha Road', vehicles: 28, congestion: 'MODERATE' });
  return (
    <div style={{ padding: '12px 12px 8px' }}>
      <SectionTitle>LIVE CAMERA FEED</SectionTitle>
      <div style={{
        background: '#060c1a', border: '1px solid #1c3050', borderRadius: 4,
        aspectRatio: '16/9', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', marginBottom: 8, position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Placeholder scanlines */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,212,132,0.015) 2px, rgba(0,212,132,0.015) 4px)',
        }} />
        <Camera size={20} style={{ color: '#1c3050', marginBottom: 6 }} />
        <span style={{ fontSize: 10, color: '#2a4060' }}>FEED PLACEHOLDER</span>
        <div style={{
          position: 'absolute', top: 6, left: 6,
          background: 'rgba(0,208,132,0.12)', border: '1px solid #00d08430',
          borderRadius: 2, padding: '2px 6px',
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          <PulseDot color="#00d084" />
          <span style={{ fontSize: 9, color: '#00d084', fontWeight: 600 }}>YOLO</span>
        </div>
        <div style={{ position: 'absolute', bottom: 6, right: 6, fontSize: 9, color: '#4a6080' }}>
          Vehicles: <span style={{ color: '#c8d8f0', fontFamily: 'JetBrains Mono, monospace' }}>{camera.vehicles}</span>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 10, color: '#4a6080' }}>{camera.location} — {camera.id}</span>
        <span style={{ fontSize: 10, color: '#ffd700' }}>{camera.congestion}</span>
      </div>
    </div>
  );
}

// ─── EXPLAINABILITY PANEL ────────────────────────────────────────────────────
function ExplainabilityPanel({ junction, allocation, officers, onClose, onViewRoute }: {
  junction: Junction; allocation: any; officers: Officer[];
  onClose: () => void; onViewRoute: () => void;
}) {
  const color = RISK_COLOR[junction.riskLevel];
  const maxFactor = Math.max(...Object.values(junction.factors) as number[]);
  const assignedOfficer = officers.find(o => o.id === allocation.officerId);
  return (
    <div
      className="slide-in"
      style={{
        width: 340, flexShrink: 0, background: '#0a1325', borderLeft: '1px solid #1e3050',
        display: 'flex', flexDirection: 'column', overflowY: 'auto',
      }}
    >
      <div style={{
        padding: '12px 14px', borderBottom: '1px solid #1c3050',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0,
      }}>
        <div>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>EXPLAINABILITY</div>
          <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>WHY THIS ASSIGNMENT?</div>
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#4a6080', cursor: 'pointer', padding: 4 }}>
          <X size={16} />
        </button>
      </div>

      <div style={{ padding: '14px', flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 10, color: '#4a6080' }}>Target</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#c8d8f0', fontWeight: 700 }}>{junction.id}</div>
            <div style={{ fontSize: 10, color: '#4a6080' }}>{junction.shortName}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <RiskBadge level={junction.riskLevel} />
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, color, fontWeight: 700, marginTop: 2 }}>
              {junction.riskScore}<span style={{ fontSize: 10, color: '#4a6080' }}>/100</span>
            </div>
          </div>
        </div>

        <Divider />

        <SectionTitle>RISK BREAKDOWN</SectionTitle>
        {[
          ['Accident History', junction.factors.accident, '#ff3a3a'],
          ['Traffic Density', junction.factors.traffic, '#ff8c42'],
          ['Pedestrian Conflict', junction.factors.pedestrian, '#ffd700'],
          ['Time of Day', junction.factors.timeOfDay, '#3b82f6'],
        ].map(([label, val, c]) => (
          <div key={label as string} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 11, color: '#7a9abf' }}>{label}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: c as string, fontWeight: 600 }}>{val}</span>
            </div>
            <MiniBar value={val as number} max={maxFactor as number + 5} color={c as string} />
          </div>
        ))}

        <Divider />

        <SectionTitle>OFFICER COMPARISON</SectionTitle>
        {[
          { id: allocation.officerId, time: allocation.responseTime, selected: true },
          ...allocation.alternatives,
        ].sort((a, b) => a.responseTime - b.responseTime).map(({ id, time, selected }: any) => {
          const o = officers.find(x => x.id === id);
          return (
            <div
              key={id}
              style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '5px 8px', marginBottom: 2, borderRadius: 3,
                background: selected ? 'rgba(37,99,235,0.1)' : 'transparent',
                border: `1px solid ${selected ? '#2563eb30' : 'transparent'}`,
              }}
            >
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: selected ? '#93c5fd' : '#4a6080' }}>
                {o?.badge ?? id}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: selected ? '#00d084' : '#4a6080' }}>
                  {time} min
                </span>
                {selected && (
                  <span style={{ fontSize: 9, color: '#3b82f6', background: 'rgba(59,130,246,0.12)', border: '1px solid #2563eb30', borderRadius: 2, padding: '1px 5px' }}>
                    ← SELECTED
                  </span>
                )}
              </div>
            </div>
          );
        })}

        <Divider />

        <SectionTitle>REASON</SectionTitle>
        <p style={{ fontSize: 11, color: '#7a9abf', lineHeight: 1.6, margin: '0 0 12px' }}>
          {allocation.reason}
        </p>

        <button onClick={onViewRoute} style={{
          width: '100%', padding: '8px 0',
          background: '#1e3a6e', border: '1px solid #2563eb',
          color: '#93c5fd', borderRadius: 3, fontSize: 11, fontWeight: 600, cursor: 'pointer',
        }}>
          VIEW DIJKSTRA ROUTE
        </button>
      </div>
    </div>
  );
}

// ─── INCIDENT MODAL ───────────────────────────────────────────────────────────
function IncidentModal({ step, form, junctions, processStep, result, timeline, onFormChange, onSimulate, onClose }: {
  step: 'form' | 'processing' | 'result';
  form: any; junctions: Junction[];
  processStep: number; result: any;
  timeline: { time: string; label: string }[];
  onFormChange: (k: string, v: any) => void;
  onSimulate: () => void; onClose: () => void;
}) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999, backdropFilter: 'blur(2px)',
    }}>
      <div className="fade-in" style={{
        background: '#0c1428', border: '1px solid #1c3050',
        borderRadius: 6, padding: '20px 24px',
        width: 460, maxWidth: '90vw', boxShadow: '0 8px 40px rgba(0,0,0,0.8)',
      }}>
        {step === 'form' && (
          <IncidentForm form={form} junctions={junctions} onChange={onFormChange} onSimulate={onSimulate} onClose={onClose} />
        )}
        {step === 'processing' && (
          <IncidentProcessing form={form} processStep={processStep} timeline={timeline} />
        )}
        {step === 'result' && result && (
          <IncidentResult result={result} junctions={junctions} onClose={onClose} />
        )}
      </div>
    </div>
  );
}

function IncidentForm({ form, junctions, onChange, onSimulate, onClose }: any) {
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 10, color: '#ff3a3a', letterSpacing: '0.1em', marginBottom: 2 }}>INCIDENT SIMULATION</div>
          <div style={{ fontSize: 14, color: '#c8d8f0', fontWeight: 700 }}>Simulate Incident</div>
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#4a6080', cursor: 'pointer' }}><X size={16} /></button>
      </div>
      <Divider />

      {[
        {
          label: 'Select Junction',
          content: (
            <select value={form.junction} onChange={e => onChange('junction', e.target.value)}
              style={selectStyle}>
              {junctions.filter((j: Junction) => j.riskLevel === 'HIGH' || j.riskLevel === 'CRITICAL').map((j: Junction) => (
                <option key={j.id} value={j.id}>{j.id} — {j.shortName}</option>
              ))}
            </select>
          ),
        },
        {
          label: 'Incident Type',
          content: (
            <select value={form.type} onChange={e => onChange('type', e.target.value)} style={selectStyle}>
              {['Accident', 'Breakdown', 'Protest', 'VIP Movement', 'Medical Emergency'].map(t => (
                <option key={t}>{t}</option>
              ))}
            </select>
          ),
        },
        {
          label: 'Severity',
          content: (
            <select value={form.severity} onChange={e => onChange('severity', e.target.value)} style={selectStyle}>
              {['High', 'Critical', 'Medium', 'Low'].map(s => <option key={s}>{s}</option>)}
            </select>
          ),
        },
        {
          label: 'Nearby Congestion Impact',
          content: (
            <button onClick={() => onChange('congestion', !form.congestion)} style={{
              ...selectStyle, cursor: 'pointer', color: form.congestion ? '#00d084' : '#4a6080',
              background: form.congestion ? 'rgba(0,208,132,0.08)' : '#111d35',
              border: `1px solid ${form.congestion ? '#00d08440' : '#1c3050'}`,
            }}>
              {form.congestion ? '✓ Enabled' : '○ Disabled'}
            </button>
          ),
        },
      ].map(({ label, content }) => (
        <div key={label} style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, color: '#4a6080', marginBottom: 4 }}>{label}</div>
          {content}
        </div>
      ))}

      <Divider />
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button onClick={onClose} style={{
          padding: '7px 16px', background: 'transparent', border: '1px solid #1c3050',
          color: '#4a6080', borderRadius: 3, fontSize: 11, cursor: 'pointer',
        }}>CANCEL</button>
        <button onClick={onSimulate} style={{
          padding: '7px 16px', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)',
          color: '#fca5a5', borderRadius: 3, fontSize: 11, fontWeight: 700, cursor: 'pointer',
          letterSpacing: '0.05em',
        }}>SIMULATE</button>
      </div>
    </>
  );
}

function IncidentProcessing({ form, processStep, timeline }: any) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: '#ff3a3a', letterSpacing: '0.1em', marginBottom: 4 }}>INCIDENT DETECTED</div>
      <div style={{ fontSize: 16, color: '#c8d8f0', fontWeight: 700, marginBottom: 4 }}>Junction {form.junction}</div>
      <div style={{ fontSize: 11, color: '#4a6080', marginBottom: 16 }}>Recalculating deployment…</div>
      <div style={{ textAlign: 'left', marginBottom: 16 }}>
        {PROCESS_STEPS.map((step, i) => {
          const done = i < processStep;
          const active = i === processStep;
          return (
            <div
              key={step}
              className={done || active ? 'step-appear' : undefined}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '6px 0', opacity: done ? 1 : active ? 0.9 : 0.3,
                animationDelay: `${i * 0.1}s`,
              }}
            >
              <span style={{ width: 16, textAlign: 'center' }}>
                {done ? <CheckCircle size={13} style={{ color: '#00d084' }} /> : active ? <Activity size={13} style={{ color: '#3b82f6' }} /> : <span style={{ fontSize: 10, color: '#2a4060' }}>○</span>}
              </span>
              <span style={{ fontSize: 11, color: done ? '#00d084' : active ? '#93c5fd' : '#2a4060' }}>{i === 0 ? 'Incident detected' : step}</span>
            </div>
          );
        })}
      </div>
      {processStep > 0 && timeline.length > 0 && (
        <div style={{ borderTop: '1px solid #1c3050', paddingTop: 10 }}>
          <div style={{ fontSize: 9, color: '#2a4060', marginBottom: 6, letterSpacing: '0.08em' }}>LIVE INCIDENT TIMELINE</div>
          {timeline.slice(-4).map((t: any, i: number) => (
            <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 4 }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, color: '#2a4060', flexShrink: 0 }}>{t.time}</span>
              <span style={{ fontSize: 10, color: '#4a6080' }}>{t.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function IncidentResult({ result, junctions, onClose }: any) {
  const junction = junctions.find((j: Junction) => j.id === result.affectedJunction);
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 10, color: '#00d084', letterSpacing: '0.1em', marginBottom: 2 }}>REDEPLOYMENT READY</div>
          <div style={{ fontSize: 14, color: '#c8d8f0', fontWeight: 700 }}>DEPLOYMENT UPDATED</div>
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#4a6080', cursor: 'pointer' }}><X size={16} /></button>
      </div>

      <div style={{ background: '#111d35', border: '1px solid #1c3050', borderRadius: 4, padding: '10px 12px', marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: '#4a6080', marginBottom: 8 }}>Junction {result.affectedJunction} — {junction?.shortName}</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 9, color: '#2a4060', marginBottom: 2 }}>BEFORE</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 18, color: '#ff8c42' }}>{result.prevRiskScore}</div>
            <div style={{ fontSize: 9, color: '#2a4060' }}>risk · {result.prevCongestion.toFixed(2)}× congestion</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 9, color: '#2a4060', marginBottom: 2 }}>AFTER</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 18, color: '#ff3a3a' }}>{result.newRiskScore}</div>
            <div style={{ fontSize: 9, color: '#2a4060' }}>risk · {result.newCongestion.toFixed(2)}× congestion</div>
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 6 }}>
          <div style={{ fontSize: 9, color: '#4a6080', letterSpacing: '0.08em' }}>BEFORE</div>
          <div style={{ fontSize: 9, color: '#4a6080', letterSpacing: '0.08em' }}>AFTER</div>
        </div>
        {(result.previousOfficers ?? []).map((r: any, i: number) => {
          const after = (result.newOfficers ?? [])[i];
          return (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 4 }}>
              <div style={{ fontSize: 11, color: '#4a6080', fontFamily: 'JetBrains Mono, monospace' }}>
                {r.officerId} → {r.fromJunction ?? '—'}
              </div>
              <div style={{ fontSize: 11, color: '#93c5fd', fontFamily: 'JetBrains Mono, monospace' }}>
                {after?.officerId ?? r.officerId} → {after?.toJunction ?? r.toJunction}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ background: 'rgba(0,208,132,0.08)', border: '1px solid rgba(0,208,132,0.2)', borderRadius: 4, padding: '10px 12px', marginBottom: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, textAlign: 'center' }}>
          <div>
            <div style={{ fontSize: 9, color: '#2a4060' }}>Response Before</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, color: '#4a6080' }}>{result.responseTimeBefore} min</div>
          </div>
          <div>
            <div style={{ fontSize: 9, color: '#2a4060' }}>Response After</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, color: '#00d084', fontWeight: 700 }}>{result.responseTimeAfter} min</div>
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: 6 }}>
          <TrendingDown size={12} style={{ color: '#00d084', display: 'inline', marginRight: 4 }} />
          <span style={{ fontSize: 11, color: '#00d084', fontWeight: 600 }}>
            ↓ {result.improvementPercentage}% improvement
          </span>
        </div>
      </div>

      <button onClick={onClose} style={{
        width: '100%', padding: '8px 0',
        background: '#1e3a6e', border: '1px solid #2563eb',
        color: '#93c5fd', borderRadius: 3, fontSize: 11, fontWeight: 600, cursor: 'pointer',
      }}>
        VIEW UPDATED MAP
      </button>
    </>
  );
}

// ─── OFFICER DETAIL MODAL ────────────────────────────────────────────────────
function OfficerDetailModal({ officer: o, junction, onClose, onLock, onUnlock, onViewRoute }: {
  officer: Officer; junction: Junction | null;
  onClose: () => void; onLock: (id: string) => void; onUnlock: (id: string) => void; onViewRoute: () => void;
}) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9000,
    }}
      onClick={onClose}
    >
      <div className="fade-in" style={{
        background: '#0c1428', border: '1px solid #1c3050', borderRadius: 5,
        padding: '16px 20px', width: 300, boxShadow: '0 4px 24px rgba(0,0,0,0.8)',
      }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#c8d8f0', fontWeight: 700 }}>{o.badge}</div>
            <div style={{ fontSize: 11, color: '#4a6080' }}>{o.name}</div>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#4a6080', cursor: 'pointer' }}><X size={14} /></button>
        </div>
        <Divider />
        {[
          ['Status', <span style={{ color: STATUS_COLOR[o.status], fontWeight: 600 }}>● {o.status}</span>],
          ['Location', o.locationName],
          ['Assigned', o.assignedJunction ?? '—'],
          ['Response', o.responseTime != null ? `${o.responseTime} min` : '—'],
          ['Distance', o.distanceKm != null ? `${o.distanceKm} km` : '—'],
        ].map(([label, val]) => (
          <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: '#4a6080' }}>{label}</span>
            <span style={{ fontSize: 11, color: '#c8d8f0' }}>{val}</span>
          </div>
        ))}
        {o.locked && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'rgba(107,114,128,0.1)', border: '1px solid rgba(107,114,128,0.2)',
            borderRadius: 3, padding: '6px 10px', marginBottom: 10,
          }}>
            <Lock size={11} style={{ color: '#6b7280' }} />
            <span style={{ fontSize: 10, color: '#4a6080' }}>Protected from automatic redeployment.</span>
          </div>
        )}
        <Divider />
        <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
          <button onClick={onViewRoute} style={{
            flex: 1, padding: '6px 0', background: '#111d35', border: '1px solid #1e3050',
            color: '#7a9abf', borderRadius: 3, fontSize: 10, cursor: 'pointer',
          }}>View Route</button>
          {o.locked ? (
            <button onClick={() => { onUnlock(o.id); onClose(); }} style={{
              flex: 1, padding: '6px 0', background: 'rgba(37,99,235,0.1)', border: '1px solid #2563eb40',
              color: '#93c5fd', borderRadius: 3, fontSize: 10, cursor: 'pointer',
            }}><Unlock size={10} style={{ display: 'inline', marginRight: 3 }} />Unlock</button>
          ) : (
            <button onClick={() => { onLock(o.id); onClose(); }} style={{
              flex: 1, padding: '6px 0', background: 'rgba(107,114,128,0.1)', border: '1px solid #6b728040',
              color: '#9ca3af', borderRadius: 3, fontSize: 10, cursor: 'pointer',
            }}><Lock size={10} style={{ display: 'inline', marginRight: 3 }} />Lock</button>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── MAP PLACEHOLDER ─────────────────────────────────────────────────────────
function MapPlaceholder() {
  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#060c1a',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#2a4060' }}>
        <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />
        <span style={{ fontSize: 12 }}>Loading Nagpur map…</span>
      </div>
    </div>
  );
}

// ─── VIEW: RISK INTELLIGENCE ─────────────────────────────────────────────────
function RiskIntelligenceView({ junctions, selectedJunction, layers, onSelectJunction, onToggleLayer }: {
  junctions: Junction[]; selectedJunction: Junction | null;
  layers: LayerState; onSelectJunction: (j: Junction) => void; onToggleLayer: (k: string) => void;
}) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const riskLayers = { ...layers, officers: false, routes: false };
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Left: map */}
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={[]} routes={[]} layers={riskLayers}
          selectedJunction={selectedJunction} activeRoute={null} incidentJunctionId={null}
          onSelectJunction={onSelectJunction} onSelectOfficer={() => {}} onViewRoute={() => {}} />
        <MapOverlayControls layers={riskLayers} onToggleLayer={onToggleLayer} />
      </div>
      {/* Right: risk table */}
      <div style={{ width: 360, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
          <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>RISK INTELLIGENCE</div>
          <div style={{ fontSize: 10, color: '#4a6080', marginTop: 2 }}>{junctions.length} junctions analysed · Nagpur City</div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {junctions.map((j, i) => {
            const color = RISK_COLOR[j.riskLevel];
            const isSelected = selectedJunction?.id === j.id;
            const isExpanded = expandedId === j.id;
            const maxFactor = Math.max(...Object.values(j.factors) as number[]);
            return (
              <div key={j.id}>
                <button
                  onClick={() => { onSelectJunction(j); setExpandedId(isExpanded ? null : j.id); }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10, width: '100%',
                    padding: '8px 14px', background: isSelected ? '#111d35' : 'transparent',
                    border: 'none', borderBottom: '1px solid #1c3050', cursor: 'pointer', textAlign: 'left',
                  }}
                >
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#2a4060', width: 18, flexShrink: 0 }}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#c8d8f0', fontWeight: 700 }}>{j.id}</span>
                      <RiskBadge level={j.riskLevel} />
                    </div>
                    <div style={{ fontSize: 10, color: '#4a6080', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{j.name}</div>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0 }}>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 20, color, fontWeight: 700, lineHeight: 1 }}>{j.riskScore}</div>
                    <div style={{ fontSize: 9, color: '#4a6080' }}>/ 100</div>
                  </div>
                  <ChevronDown size={10} style={{ color: '#4a6080', flexShrink: 0, transform: isExpanded ? 'rotate(180deg)' : undefined, transition: 'transform 0.15s' }} />
                </button>
                {isExpanded && (
                  <div style={{ padding: '10px 14px 12px', background: '#0a1325', borderBottom: '1px solid #1c3050' }}>
                    <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.08em', marginBottom: 8 }}>RISK FACTOR BREAKDOWN</div>
                    {([
                      ['Accident History', j.factors.accident, '#ff3a3a'],
                      ['Traffic Density', j.factors.traffic, '#ff8c42'],
                      ['Pedestrian Conflict', j.factors.pedestrian, '#ffd700'],
                      ['Time of Day', j.factors.timeOfDay, '#3b82f6'],
                    ] as [string, number, string][]).map(([label, val, c]) => (
                      <div key={label} style={{ marginBottom: 7 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                          <span style={{ fontSize: 10, color: '#7a9abf' }}>{label}</span>
                          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: c, fontWeight: 600 }}>{val}</span>
                        </div>
                        <MiniBar value={val} max={maxFactor + 5} color={c} />
                      </div>
                    ))}
                    <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                      <div>
                        <div style={{ fontSize: 9, color: '#2a4060' }}>Congestion</div>
                        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#ffd700' }}>{j.congestionFactor.toFixed(2)}×</div>
                      </div>
                      {j.assignedOfficer && (
                        <div>
                          <div style={{ fontSize: 9, color: '#2a4060' }}>Assigned Officer</div>
                          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#00d084' }}>{j.assignedOfficer}</div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: LIVE TRAFFIC ───────────────────────────────────────────────────────
function LiveTrafficView({ traffic, junctions, layers, onSelectJunction, onToggleLayer }: {
  traffic: import('./types/api').TrafficData; junctions: Junction[]; layers: LayerState;
  onSelectJunction: (j: Junction) => void; onToggleLayer: (k: string) => void;
}) {
  const trendMax = Math.max(...traffic.trend);
  const trafficLayers = { ...layers, officers: false, routes: false };
  const sorted = [...junctions].sort((a, b) => b.congestionFactor - a.congestionFactor);
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={[]} routes={[]} layers={trafficLayers}
          selectedJunction={null} activeRoute={null} incidentJunctionId={null}
          onSelectJunction={onSelectJunction} onSelectOfficer={() => {}} onViewRoute={() => {}} />
        <MapOverlayControls layers={trafficLayers} onToggleLayer={onToggleLayer} />
      </div>
      <div style={{ width: 360, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
              <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>LIVE TRAFFIC</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <PulseDot color="#00d084" />
              <span style={{ fontSize: 10, color: '#00d084', fontWeight: 600 }}>LIVE</span>
            </div>
          </div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1, padding: '12px 14px' }}>
          {/* Stats grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
            {[
              { label: 'Vehicle Count', value: traffic.vehicleCount.toLocaleString(), color: '#c8d8f0' },
              { label: 'Avg Density', value: traffic.avgDensity.toFixed(2), color: '#c8d8f0' },
              { label: 'Congestion Level', value: traffic.congestionLevel, color: '#ffd700' },
              { label: 'Congestion Factor', value: `${traffic.congestionFactor}×`, color: '#ffd700' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: '#111d35', border: '1px solid #1c3050', borderRadius: 4, padding: '8px 10px' }}>
                <div style={{ fontSize: 9, color: '#4a6080', marginBottom: 4 }}>{label}</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color, fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </div>
          {/* Trend chart */}
          <SectionTitle>TRAFFIC TREND</SectionTitle>
          <div style={{ height: 60, display: 'flex', alignItems: 'flex-end', gap: 2, marginBottom: 14 }}>
            {traffic.trend.map((v: number, i: number) => (
              <div key={i} style={{
                flex: 1, background: '#2563eb',
                height: `${(v / trendMax) * 100}%`,
                borderRadius: '2px 2px 0 0',
                opacity: 0.4 + (i / traffic.trend.length) * 0.6, minHeight: 2,
              }} />
            ))}
          </div>
          {/* Junction traffic density */}
          <SectionTitle>JUNCTION CONGESTION</SectionTitle>
          {sorted.map(j => (
            <button key={j.id} onClick={() => onSelectJunction(j)} style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '6px 8px', marginBottom: 2, borderRadius: 3,
              background: 'transparent', border: '1px solid transparent', cursor: 'pointer', textAlign: 'left',
            }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', width: 40, flexShrink: 0 }}>{j.id}</span>
              <div style={{ flex: 1 }}>
                <div style={{ height: 4, background: '#1c3050', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, j.congestionFactor * 50)}%`, height: '100%', background: RISK_COLOR[j.riskLevel], borderRadius: 2 }} />
                </div>
              </div>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#ffd700', width: 36, textAlign: 'right', flexShrink: 0 }}>{j.congestionFactor.toFixed(2)}×</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: OFFICERS ───────────────────────────────────────────────────────────
function OfficersView({ officers, junctions, selectedOfficer, layers, onSelectOfficer, onLock, onUnlock, onToggleLayer }: {
  officers: Officer[]; junctions: Junction[]; selectedOfficer: Officer | null;
  layers: LayerState; onSelectOfficer: (o: Officer) => void;
  onLock: (id: string) => void; onUnlock: (id: string) => void; onToggleLayer: (k: string) => void;
}) {
  const officerLayers = { ...layers, risk: false, routes: false };
  const statusGroups = ['AVAILABLE', 'ASSIGNED', 'LOCKED', 'OFFLINE'] as const;
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={officers} routes={[]} layers={officerLayers}
          selectedJunction={null} activeRoute={null} incidentJunctionId={null}
          onSelectJunction={() => {}} onSelectOfficer={onSelectOfficer} onViewRoute={() => {}} />
        <MapOverlayControls layers={officerLayers} onToggleLayer={onToggleLayer} />
      </div>
      <div style={{ width: 380, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
          <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>OFFICERS</div>
          <div style={{ display: 'flex', gap: 12, marginTop: 6 }}>
            {statusGroups.map(s => {
              const count = officers.filter(o => o.status === s).length;
              return count > 0 ? (
                <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: STATUS_COLOR[s], display: 'inline-block' }} />
                  <span style={{ fontSize: 10, color: '#4a6080' }}>{count} {s.toLowerCase()}</span>
                </div>
              ) : null;
            })}
          </div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {officers.map(o => {
            const isSelected = selectedOfficer?.id === o.id;
            const junc = junctions.find(j => j.id === o.assignedJunction);
            return (
              <div key={o.id} style={{
                padding: '10px 14px', borderBottom: '1px solid #1c3050',
                background: isSelected ? '#111d35' : 'transparent', cursor: 'pointer',
              }}
                onClick={() => onSelectOfficer(o)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <PulseDot color={STATUS_COLOR[o.status]} />
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#c8d8f0', fontWeight: 700 }}>{o.badge}</span>
                        {o.locked && <Lock size={10} style={{ color: '#6b7280' }} />}
                      </div>
                      <div style={{ fontSize: 10, color: '#4a6080' }}>{o.name}</div>
                    </div>
                  </div>
                  <span style={{
                    fontSize: 9, color: STATUS_COLOR[o.status], background: `${STATUS_COLOR[o.status]}18`,
                    border: `1px solid ${STATUS_COLOR[o.status]}30`, borderRadius: 2, padding: '2px 6px', fontWeight: 600,
                  }}>{o.status}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4 }}>
                  <div>
                    <div style={{ fontSize: 9, color: '#2a4060' }}>Location</div>
                    <div style={{ fontSize: 10, color: '#7a9abf' }}>{o.locationName}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: '#2a4060' }}>Assigned</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: junc ? RISK_COLOR[junc.riskLevel] : '#4a6080' }}>{o.assignedJunction ?? '—'}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: '#2a4060' }}>Response</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#3b82f6' }}>{o.responseTime != null ? `${o.responseTime}m` : '—'}</div>
                  </div>
                </div>
                {isSelected && (
                  <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                    {o.locked ? (
                      <button onClick={e => { e.stopPropagation(); onUnlock(o.id); }} style={{ flex: 1, padding: '5px 0', background: 'rgba(37,99,235,0.1)', border: '1px solid #2563eb40', color: '#93c5fd', borderRadius: 3, fontSize: 10, cursor: 'pointer' }}>
                        <Unlock size={9} style={{ display: 'inline', marginRight: 3 }} />Unlock
                      </button>
                    ) : (
                      <button onClick={e => { e.stopPropagation(); onLock(o.id); }} style={{ flex: 1, padding: '5px 0', background: 'rgba(107,114,128,0.1)', border: '1px solid #6b728040', color: '#9ca3af', borderRadius: 3, fontSize: 10, cursor: 'pointer' }}>
                        <Lock size={9} style={{ display: 'inline', marginRight: 3 }} />Lock
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: DEPLOYMENT ─────────────────────────────────────────────────────────
function DeploymentView({ allocations, officers, junctions, routes, activeRoute, selectedJunction, layers, onSelectJunction, onViewRoute, onToggleLayer }: {
  allocations: OfficerAllocation[]; officers: Officer[]; junctions: Junction[];
  routes: Route[]; activeRoute: Route | null; selectedJunction: Junction | null;
  layers: LayerState; onSelectJunction: (j: Junction) => void;
  onViewRoute: (j: Junction) => void; onToggleLayer: (k: string) => void;
}) {
  const deployLayers = { ...layers, traffic: false };
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={officers} routes={routes} layers={deployLayers}
          selectedJunction={selectedJunction} activeRoute={activeRoute} incidentJunctionId={null}
          onSelectJunction={onSelectJunction} onSelectOfficer={() => {}} onViewRoute={onViewRoute} />
        <MapOverlayControls layers={deployLayers} onToggleLayer={onToggleLayer} />
      </div>
      <div style={{ width: 380, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
          <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>DEPLOYMENT</div>
          <div style={{ fontSize: 10, color: '#4a6080', marginTop: 2 }}>{allocations.length} active allocations · Dijkstra routing</div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {allocations.map(a => {
            const officer = officers.find(o => o.id === a.officerId);
            const junction = junctions.find(j => j.id === a.junctionId);
            if (!officer || !junction) return null;
            const isSelected = selectedJunction?.id === junction.id;
            return (
              <div key={`${a.officerId}-${a.junctionId}`}
                onClick={() => onSelectJunction(junction)}
                style={{
                  display: 'flex', gap: 10, width: '100%', padding: '10px 14px',
                  borderBottom: '1px solid #1c3050', background: isSelected ? '#111d35' : 'transparent',
                  cursor: 'pointer',
                }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#00d084', fontWeight: 600 }}>{officer.badge}</span>
                    <span style={{ fontSize: 10, color: '#4a6080' }}>→</span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', fontWeight: 700 }}>{junction.id}</span>
                    <RiskBadge level={junction.riskLevel} />
                  </div>
                  <div style={{ fontSize: 10, color: '#4a6080', marginBottom: 4 }}>{junction.shortName}</div>
                  <div style={{ display: 'flex', gap: 16 }}>
                    <div>
                      <span style={{ fontSize: 9, color: '#2a4060' }}>Response </span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#3b82f6' }}>{a.responseTime}m</span>
                    </div>
                    <div>
                      <span style={{ fontSize: 9, color: '#2a4060' }}>Distance </span>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#7a9abf' }}>{a.distanceKm}km</span>
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 4 }}>
                  <button
                    onClick={e => { e.stopPropagation(); onViewRoute(junction); }}
                    style={{ padding: '4px 8px', background: '#0f2a1a', border: '1px solid #166534', color: '#4ade80', borderRadius: 3, fontSize: 9, cursor: 'pointer', whiteSpace: 'nowrap' }}
                  >
                    ROUTE
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: INCIDENTS ──────────────────────────────────────────────────────────
const MOCK_INCIDENTS = [
  { id: 'INC-001', junction: 'J007', type: 'Accident', severity: 'HIGH', time: '10:24 AM', status: 'ACTIVE' },
  { id: 'INC-002', junction: 'J003', type: 'Breakdown', severity: 'MEDIUM', time: '09:51 AM', status: 'RESOLVED' },
  { id: 'INC-003', junction: 'J012', type: 'VIP Movement', severity: 'HIGH', time: '09:30 AM', status: 'RESOLVED' },
  { id: 'INC-004', junction: 'J001', type: 'Protest', severity: 'CRITICAL', time: '08:45 AM', status: 'RESOLVED' },
  { id: 'INC-005', junction: 'J018', type: 'Medical Emergency', severity: 'HIGH', time: '08:12 AM', status: 'RESOLVED' },
];

function IncidentsView({ junctions, incidentResult, incidentJunctionId, layers, onSimulate, onSelectJunction, onToggleLayer }: {
  junctions: Junction[]; incidentResult: any; incidentJunctionId: string | null;
  layers: LayerState; onSimulate: () => void;
  onSelectJunction: (j: Junction) => void; onToggleLayer: (k: string) => void;
}) {
  const incLayers = { ...layers, routes: false };
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={[]} routes={[]} layers={incLayers}
          selectedJunction={incidentJunctionId ? (junctions.find(j => j.id === incidentJunctionId) ?? null) : null}
          activeRoute={null} incidentJunctionId={incidentJunctionId}
          onSelectJunction={onSelectJunction} onSelectOfficer={() => {}} onViewRoute={() => {}} />
        <MapOverlayControls layers={incLayers} onToggleLayer={onToggleLayer} />
      </div>
      <div style={{ width: 380, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
              <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>INCIDENTS</div>
            </div>
            <button onClick={onSimulate} style={{
              display: 'flex', alignItems: 'center', gap: 5,
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.35)',
              color: '#fca5a5', borderRadius: 3, padding: '5px 10px',
              fontSize: 10, fontWeight: 600, cursor: 'pointer',
            }}>
              <AlertCircle size={10} />
              SIMULATE
            </button>
          </div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1, padding: '8px 0' }}>
          {/* Last simulation result */}
          {incidentResult && (
            <div style={{ margin: '8px 14px 4px', background: 'rgba(0,208,132,0.06)', border: '1px solid rgba(0,208,132,0.2)', borderRadius: 4, padding: '10px 12px' }}>
              <div style={{ fontSize: 10, color: '#00d084', fontWeight: 600, letterSpacing: '0.08em', marginBottom: 6 }}>LAST SIMULATION RESULT</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Junction</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0' }}>{incidentResult.affectedJunction}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Improvement</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#00d084' }}>
                    ↓{Math.round((1 - incidentResult.newAvgResponse / incidentResult.prevAvgResponse) * 100)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Risk Before</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#ff8c42' }}>{incidentResult.prevRiskScore}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Risk After</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#ff3a3a' }}>{incidentResult.newRiskScore}</div>
                </div>
              </div>
            </div>
          )}
          {/* Incident list */}
          <div style={{ padding: '10px 14px 4px' }}>
            <SectionTitle>RECENT INCIDENTS</SectionTitle>
          </div>
          {MOCK_INCIDENTS.map(inc => {
            const isActive = inc.status === 'ACTIVE';
            const sevColor = inc.severity === 'CRITICAL' ? '#ff3a3a' : inc.severity === 'HIGH' ? '#ff8c42' : '#ffd700';
            const j = junctions.find(x => x.id === inc.junction);
            return (
              <button key={inc.id} onClick={() => j && onSelectJunction(j)} style={{
                display: 'flex', gap: 10, width: '100%', padding: '9px 14px',
                borderBottom: '1px solid #1c3050', background: isActive ? 'rgba(239,68,68,0.04)' : 'transparent',
                border: 'none', borderBottomColor: '#1c3050', borderBottomWidth: 1, borderBottomStyle: 'solid',
                cursor: 'pointer', textAlign: 'left',
              }}>
                <div style={{ width: 4, borderRadius: 2, background: isActive ? sevColor : '#2a4060', flexShrink: 0, alignSelf: 'stretch' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#4a6080' }}>{inc.id}</span>
                    <span style={{ fontSize: 9, color: isActive ? '#ff3a3a' : '#2a4060', background: isActive ? 'rgba(239,68,68,0.1)' : '#111d35', border: `1px solid ${isActive ? 'rgba(239,68,68,0.2)' : '#1c3050'}`, borderRadius: 2, padding: '1px 5px' }}>{inc.status}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 2 }}>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', fontWeight: 600 }}>{inc.junction}</span>
                    <span style={{ fontSize: 11, color: '#7a9abf' }}>{inc.type}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <span style={{ fontSize: 9, color: sevColor, fontWeight: 600 }}>{inc.severity}</span>
                    <span style={{ fontSize: 9, color: '#2a4060' }}>{inc.time}</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: COVERAGE ───────────────────────────────────────────────────────────
function CoverageView({ coverage, junctions, officers, allocations, uncoveredJunctions, layers, onSelectJunction, onToggleLayer }: {
  coverage: any; junctions: Junction[]; officers: Officer[]; allocations: OfficerAllocation[];
  uncoveredJunctions: Junction[]; layers: LayerState;
  onSelectJunction: (j: Junction) => void; onToggleLayer: (k: string) => void;
}) {
  const covLayers = { ...layers, traffic: false, routes: false };
  const pct = coverage.coveragePct;
  const circumference = 2 * Math.PI * 36;
  const offset = circumference * (1 - pct / 100);
  const coveredJunctions = junctions.filter(j => !coverage.uncoveredJunctions.includes(j.id));
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <MapView junctions={junctions} officers={officers} routes={[]} layers={covLayers}
          selectedJunction={null} activeRoute={null} incidentJunctionId={null}
          onSelectJunction={onSelectJunction} onSelectOfficer={() => {}} onViewRoute={() => {}} />
        <MapOverlayControls layers={covLayers} onToggleLayer={onToggleLayer} />
      </div>
      <div style={{ width: 380, flexShrink: 0, background: '#0c1428', borderLeft: '1px solid #1c3050', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid #1c3050', flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 2 }}>NOVAROUTE.AI</div>
          <div style={{ fontSize: 13, color: '#c8d8f0', fontWeight: 700 }}>COVERAGE</div>
        </div>
        <div style={{ overflowY: 'auto', flex: 1, padding: '14px' }}>
          {/* Donut + stats */}
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 16 }}>
            <svg width={86} height={86} style={{ flexShrink: 0 }}>
              <circle cx={43} cy={43} r={36} fill="none" stroke="#1c3050" strokeWidth={6} />
              <circle cx={43} cy={43} r={36} fill="none" stroke="#00c896" strokeWidth={6}
                strokeDasharray={circumference} strokeDashoffset={offset}
                strokeLinecap="round" transform="rotate(-90 43 43)" />
              <text x={43} y={48} textAnchor="middle" style={{ fill: '#c8d8f0', fontFamily: 'JetBrains Mono, monospace', fontSize: 16, fontWeight: 700 }}>{pct}%</text>
            </svg>
            <div>
              <div style={{ fontSize: 12, color: '#c8d8f0', fontWeight: 600, marginBottom: 6 }}>{coverage.coveredJunctions} / {coverage.totalJunctions} junctions covered</div>
              <div style={{ fontSize: 10, color: '#4a6080', marginBottom: 8 }}>within {coverage.thresholdMinutes} minute response</div>
              <div style={{ display: 'flex', gap: 14 }}>
                {[
                  { label: 'Avg', value: `${coverage.avgResponse}m`, color: '#3b82f6' },
                  { label: 'Worst', value: `${coverage.worstResponse}m`, color: '#ff8c42' },
                  { label: 'Uncov.', value: String(coverage.uncoveredCount), color: '#ff3a3a' },
                ].map(({ label, value, color }) => (
                  <div key={label}>
                    <div style={{ fontSize: 9, color: '#2a4060' }}>{label}</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color, fontWeight: 700 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <Divider />
          {/* Officer coverage rows */}
          <SectionTitle>OFFICER COVERAGE</SectionTitle>
          {officers.filter(o => o.status !== 'OFFLINE').map(o => {
            const alloc = allocations.find(a => a.officerId === o.id);
            const j = alloc ? junctions.find(x => x.id === alloc.junctionId) : null;
            return (
              <div key={o.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0', borderBottom: '1px solid #0f1e38' }}>
                <PulseDot color={STATUS_COLOR[o.status]} />
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', width: 60 }}>{o.badge}</span>
                <span style={{ flex: 1, fontSize: 10, color: '#4a6080' }}>{j ? `${j.id} — ${j.shortName}` : 'Unassigned'}</span>
                {alloc && <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#3b82f6' }}>{alloc.responseTime}m</span>}
              </div>
            );
          })}
          <Divider />
          {/* Uncovered junctions */}
          {uncoveredJunctions.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 8 }}>
                <AlertTriangle size={11} style={{ color: '#f59e0b' }} />
                <SectionTitle>{uncoveredJunctions.length} UNCOVERED JUNCTIONS</SectionTitle>
              </div>
              {uncoveredJunctions.map(j => (
                <button key={j.id} onClick={() => onSelectJunction(j)} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  width: '100%', padding: '5px 8px', marginBottom: 3,
                  background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)',
                  borderRadius: 3, cursor: 'pointer',
                }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#f59e0b' }}>{j.id}</span>
                  <span style={{ fontSize: 10, color: '#4a6080' }}>{j.shortName}</span>
                  <RiskBadge level={j.riskLevel} />
                </button>
              ))}
            </>
          )}
          <Divider />
          <SectionTitle>COVERED JUNCTIONS</SectionTitle>
          {coveredJunctions.map(j => (
            <button key={j.id} onClick={() => onSelectJunction(j)} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              width: '100%', padding: '4px 8px', marginBottom: 2,
              background: 'transparent', border: '1px solid transparent', borderRadius: 3, cursor: 'pointer',
            }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#4a6080' }}>{j.id}</span>
              <span style={{ fontSize: 10, color: '#2a4060' }}>{j.shortName}</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <CheckCircle size={9} style={{ color: '#00c896' }} />
                <span style={{ fontSize: 9, color: '#00c896' }}>COVERED</span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── VIEW: PERFORMANCE ────────────────────────────────────────────────────────
function PerformanceView({ baseline, coverage, traffic, officers, junctions, allocations }: {
  baseline: any; coverage: any; traffic: any;
  officers: Officer[]; junctions: Junction[]; allocations: OfficerAllocation[];
}) {
  const avgResponseTime = allocations.length
    ? (allocations.reduce((s, a) => s + a.responseTime, 0) / allocations.length).toFixed(1)
    : '—';
  const activeOfficers = officers.filter(o => o.status !== 'OFFLINE').length;
  const criticalCount = junctions.filter(j => j.riskLevel === 'CRITICAL').length;
  const improvement = baseline
    ? Math.round((1 - baseline.novaroute.avgResponse / baseline.static.avgResponse) * 100)
    : 0;
  const trendMax = traffic ? Math.max(...traffic.trend) : 1;

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', background: '#070d1a' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 10, color: '#4a6080', letterSpacing: '0.1em', marginBottom: 4 }}>NOVAROUTE.AI</div>
          <div style={{ fontSize: 16, color: '#c8d8f0', fontWeight: 700 }}>PERFORMANCE OVERVIEW</div>
          <div style={{ fontSize: 10, color: '#4a6080', marginTop: 2 }}>Operational metrics · Nagpur City</div>
        </div>

        {/* Top KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 20 }}>
          {[
            { label: 'AVG RESPONSE TIME', value: `${avgResponseTime}m`, sub: `↓ ${improvement}% vs static`, color: '#3b82f6', icon: <Activity size={14} /> },
            { label: 'COVERAGE', value: `${coverage?.coveragePct ?? '—'}%`, sub: `${coverage?.coveredJunctions ?? '—'} / ${coverage?.totalJunctions ?? '—'} junctions`, color: '#00c896', icon: <Eye size={14} /> },
            { label: 'ACTIVE OFFICERS', value: `${activeOfficers} / ${officers.length}`, sub: `${officers.filter(o => o.locked).length} locked`, color: '#00d084', icon: <Shield size={14} /> },
            { label: 'CRITICAL RISK', value: String(criticalCount), sub: `${junctions.filter(j => j.riskLevel === 'HIGH').length} HIGH risk also`, color: '#ff3a3a', icon: <AlertTriangle size={14} /> },
          ].map(({ label, value, sub, color, icon }) => (
            <div key={label} style={{ background: '#0c1428', border: '1px solid #1c3050', borderRadius: 4, padding: '12px 14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <span style={{ fontSize: 9, color: '#4a6080', letterSpacing: '0.1em', fontWeight: 600 }}>{label}</span>
                <span style={{ color, opacity: 0.7 }}>{icon}</span>
              </div>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 24, fontWeight: 700, color, lineHeight: 1, marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 10, color: '#4a6080' }}>{sub}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Baseline comparison */}
          {baseline && (
            <div style={{ background: '#0c1428', border: '1px solid #1c3050', borderRadius: 4, padding: '14px' }}>
              <SectionTitle>NOVAROUTE VS STATIC BASELINE</SectionTitle>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px 80px', gap: '4px 0' }}>
                <div />
                <div style={{ fontSize: 9, color: '#4a6080', textAlign: 'center', letterSpacing: '0.06em', paddingBottom: 6 }}>STATIC</div>
                <div style={{ fontSize: 9, color: '#3b82f6', textAlign: 'center', letterSpacing: '0.06em', paddingBottom: 6 }}>NOVAROUTE</div>
                {[
                  { label: 'Avg Response', sVal: `${baseline.static.avgResponse}m`, nVal: `${baseline.novaroute.avgResponse}m`, better: baseline.novaroute.avgResponse < baseline.static.avgResponse },
                  { label: 'Coverage %', sVal: `${baseline.static.coveragePct}%`, nVal: `${baseline.novaroute.coveragePct}%`, better: baseline.novaroute.coveragePct > baseline.static.coveragePct },
                  { label: 'Covered', sVal: String(baseline.static.coveredCount), nVal: String(baseline.novaroute.coveredCount), better: baseline.novaroute.coveredCount > baseline.static.coveredCount },
                ].map(({ label, sVal, nVal, better }) => [
                  <div key={`${label}-l`} style={{ fontSize: 10, color: '#4a6080', padding: '4px 0' }}>{label}</div>,
                  <div key={`${label}-s`} style={{ textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#4a6080', padding: '4px 0' }}>{sVal}</div>,
                  <div key={`${label}-n`} style={{ textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: better ? '#00d084' : '#ff3a3a', fontWeight: 600, padding: '4px 0' }}>{nVal}</div>,
                ]).flat()}
              </div>
              <div style={{ marginTop: 10, background: 'rgba(37,99,235,0.08)', border: '1px solid #2563eb30', borderRadius: 3, padding: '8px 10px', textAlign: 'center' }}>
                <TrendingDown size={12} style={{ color: '#00d084', display: 'inline', marginRight: 4 }} />
                <span style={{ fontSize: 11, color: '#00d084', fontWeight: 600 }}>↓ {improvement}% response time improvement</span>
              </div>
            </div>
          )}

          {/* Traffic trend */}
          {traffic && (
            <div style={{ background: '#0c1428', border: '1px solid #1c3050', borderRadius: 4, padding: '14px' }}>
              <SectionTitle>NETWORK TRAFFIC TREND</SectionTitle>
              <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Congestion</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#ffd700', fontWeight: 700 }}>{traffic.congestionLevel}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Factor</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#ffd700', fontWeight: 700 }}>{traffic.congestionFactor}×</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: '#2a4060' }}>Vehicles</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#c8d8f0', fontWeight: 700 }}>{traffic.vehicleCount.toLocaleString()}</div>
                </div>
              </div>
              <div style={{ height: 80, display: 'flex', alignItems: 'flex-end', gap: 2 }}>
                {traffic.trend.map((v: number, i: number) => (
                  <div key={i} style={{
                    flex: 1, background: '#2563eb', borderRadius: '2px 2px 0 0',
                    height: `${(v / trendMax) * 100}%`, minHeight: 2,
                    opacity: 0.4 + (i / traffic.trend.length) * 0.6,
                  }} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Deployment efficiency table */}
        <div style={{ background: '#0c1428', border: '1px solid #1c3050', borderRadius: 4, padding: '14px' }}>
          <SectionTitle>DEPLOYMENT EFFICIENCY BY JUNCTION</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: '60px 1fr 80px 80px 80px 80px', gap: '0 8px', alignItems: 'center' }}>
            {['ID', 'Junction', 'Risk', 'Score', 'Response', 'Congestion'].map(h => (
              <div key={h} style={{ fontSize: 9, color: '#2a4060', letterSpacing: '0.06em', paddingBottom: 6, borderBottom: '1px solid #1c3050', marginBottom: 2 }}>{h}</div>
            ))}
            {[...junctions].sort((a, b) => b.riskScore - a.riskScore).slice(0, 10).map(j => {
              const alloc = allocations.find(a => a.junctionId === j.id);
              return [
                <div key={`${j.id}-id`} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#c8d8f0', padding: '4px 0' }}>{j.id}</div>,
                <div key={`${j.id}-name`} style={{ fontSize: 10, color: '#4a6080', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '4px 0' }}>{j.shortName}</div>,
                <div key={`${j.id}-risk`} style={{ padding: '4px 0' }}><RiskBadge level={j.riskLevel} /></div>,
                <div key={`${j.id}-score`} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: RISK_COLOR[j.riskLevel], fontWeight: 600, padding: '4px 0' }}>{j.riskScore}</div>,
                <div key={`${j.id}-rt`} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#3b82f6', padding: '4px 0' }}>{alloc ? `${alloc.responseTime}m` : '—'}</div>,
                <div key={`${j.id}-cong`} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#ffd700', padding: '4px 0' }}>{j.congestionFactor.toFixed(2)}×</div>,
              ];
            }).flat()}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── utils ────────────────────────────────────────────────────────────────────
const selectStyle: React.CSSProperties = {
  display: 'block', width: '100%', padding: '6px 10px',
  background: '#111d35', border: '1px solid #1c3050',
  color: '#c8d8f0', borderRadius: 3, fontSize: 11,
  fontFamily: 'Inter, sans-serif', appearance: 'none',
};
