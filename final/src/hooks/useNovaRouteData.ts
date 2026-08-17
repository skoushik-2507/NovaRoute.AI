import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Junction, Officer, OfficerAllocation, Route,
  TrafficData, CoverageData, BaselineData,
} from '../types/api';
import {
  getJunctions, getOfficers, getAllocations, getAllRoutes,
  getTrafficData, getCoverage, getBaseline,
} from '../services/api';

export type LoadStatus = 'idle' | 'loading' | 'success' | 'error';
export type ConnectionMode = 'demo' | 'live' | 'fallback';

export interface NovaRouteData {
  junctions:   Junction[];
  officers:    Officer[];
  allocations: OfficerAllocation[];
  routes:      Route[];
  traffic:     TrafficData | null;
  coverage:    CoverageData | null;
  baseline:    BaselineData | null;
}

export interface UseNovaRouteDataResult {
  data: NovaRouteData;
  status: LoadStatus;
  error: string | null;
  lastUpdated: Date | null;
  isBackendOnline: boolean;
  connectionMode: ConnectionMode;
  refresh: () => void;
  setJunctions:   React.Dispatch<React.SetStateAction<Junction[]>>;
  setOfficers:    React.Dispatch<React.SetStateAction<Officer[]>>;
  setRoutes:      React.Dispatch<React.SetStateAction<Route[]>>;
  setTraffic:     React.Dispatch<React.SetStateAction<TrafficData | null>>;
  setCoverage:    React.Dispatch<React.SetStateAction<CoverageData | null>>;
  setBaseline:    React.Dispatch<React.SetStateAction<BaselineData | null>>;
  setAllocations: React.Dispatch<React.SetStateAction<OfficerAllocation[]>>;
}

export function useNovaRouteData(demoMode: boolean): UseNovaRouteDataResult {
  const [junctions,   setJunctions]   = useState<Junction[]>([]);
  const [officers,    setOfficers]     = useState<Officer[]>([]);
  const [allocations, setAllocations] = useState<OfficerAllocation[]>([]);
  const [routes,      setRoutes]       = useState<Route[]>([]);
  const [traffic,     setTraffic]      = useState<TrafficData | null>(null);
  const [coverage,    setCoverage]     = useState<CoverageData | null>(null);
  const [baseline,    setBaseline]     = useState<BaselineData | null>(null);

  const [status,          setStatus]          = useState<LoadStatus>('idle');
  const [error,           setError]           = useState<string | null>(null);
  const [lastUpdated,     setLastUpdated]      = useState<Date | null>(null);
  const [isBackendOnline, setIsBackendOnline]  = useState(true);
  const [connectionMode,  setConnectionMode]   = useState<ConnectionMode>('demo');

  const loadingRef = useRef(false);

  const load = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setStatus('loading');
    setError(null);

    if (demoMode) {
      setConnectionMode('demo');
    }

    try {
      const [j, o, a, r, t, c, b] = await Promise.all([
        getJunctions(demoMode),
        getOfficers(demoMode),
        getAllocations(demoMode),
        getAllRoutes(demoMode),
        getTrafficData(demoMode),
        getCoverage(demoMode),
        getBaseline(demoMode),
      ]);
      setJunctions(j);
      setOfficers(o);
      setAllocations(a);
      setRoutes(r);
      setTraffic(t);
      setCoverage(c);
      setBaseline(b);
      setStatus('success');
      setLastUpdated(new Date());
      setIsBackendOnline(true);
      setConnectionMode(demoMode ? 'demo' : 'live');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
      setStatus('error');
      setIsBackendOnline(false);
      setConnectionMode('fallback');
      if (junctions.length === 0) {
        const [j, o, a, r, t, c, b] = await Promise.all([
          getJunctions(true),
          getOfficers(true),
          getAllocations(true),
          getAllRoutes(true),
          getTrafficData(true),
          getCoverage(true),
          getBaseline(true),
        ]);
        setJunctions(j); setOfficers(o); setAllocations(a);
        setRoutes(r); setTraffic(t); setCoverage(c); setBaseline(b);
      }
    } finally {
      loadingRef.current = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoMode]);

  useEffect(() => { load(); }, [load]);

  return {
    data: { junctions, officers, allocations, routes, traffic, coverage, baseline },
    status,
    error,
    lastUpdated,
    isBackendOnline,
    connectionMode,
    refresh: load,
    setJunctions,
    setOfficers,
    setRoutes,
    setTraffic,
    setCoverage,
    setBaseline,
    setAllocations,
  };
}
