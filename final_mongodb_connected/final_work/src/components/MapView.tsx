import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap, CircleMarker, Polyline, Popup, Tooltip } from 'react-leaflet';
import type { Junction, Officer, Route } from '../types/api';

export interface LayerState {
  risk: boolean;
  officers: boolean;
  routes: boolean;
  traffic: boolean;
  incidents: boolean;
}

interface Props {
  junctions: Junction[];
  officers: Officer[];
  routes: Route[];
  layers: LayerState;
  selectedJunction: Junction | null;
  activeRoute: Route | null;
  incidentJunctionId: string | null;
  onSelectJunction: (j: Junction) => void;
  onSelectOfficer: (o: Officer) => void;
  onViewRoute: (j: Junction) => void;
}

const RISK_COLORS: Record<string, string> = {
  LOW: '#00c896', MEDIUM: '#ffd700', HIGH: '#ff8c42', CRITICAL: '#ff3a3a',
};

const RISK_FILL: Record<string, string> = {
  LOW: 'rgba(0,200,150,0.15)', MEDIUM: 'rgba(255,215,0,0.15)',
  HIGH: 'rgba(255,140,66,0.18)', CRITICAL: 'rgba(255,58,58,0.2)',
};

function riskRadius(score: number) {
  if (score >= 76) return 14;
  if (score >= 56) return 12;
  if (score >= 31) return 10;
  return 8;
}

// Resolve canonical latitude/longitude from a Junction (backend) or a legacy
// record that might still use lat/lng. Guards against future field renames.
function jLatLng(j: Junction): [number, number] {
  return [j.latitude, j.longitude];
}

function oLatLng(o: Officer): [number, number] {
  return [o.latitude, o.longitude];
}

// ── MapController — fly-to on junction selection ──────────────────────────────
function MapController({ selectedJunction }: { selectedJunction: Junction | null }) {
  const map = useMap();
  useEffect(() => {
    if (selectedJunction) {
      map.flyTo(jLatLng(selectedJunction), 15, { duration: 1.2 });
    }
  }, [selectedJunction, map]);
  return null;
}

// ── OfficerMarkers — custom L.divIcon markers, synced with state ──────────────
function OfficerMarkers({ officers, onSelectOfficer, visible }: {
  officers: Officer[];
  onSelectOfficer: (o: Officer) => void;
  visible: boolean;
}) {
  const map = useMap();
  const markersRef = useRef<L.Marker[]>([]);

  useEffect(() => {
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    if (!visible) return;

    officers
      .filter(o => o.status !== 'OFFLINE')
      .forEach(officer => {
        const dotColor = officer.locked
          ? '#9ca3af'
          : officer.status === 'AVAILABLE' ? '#00d084' : '#f59e0b';
        const borderColor = officer.locked
          ? '#9ca3af'
          : officer.status === 'AVAILABLE' ? '#00e896' : '#fbbf24';
        const shadow = officer.locked
          ? '0 1px 4px rgba(0,0,0,0.8)'
          : officer.status === 'AVAILABLE'
            ? '0 1px 6px rgba(0,216,132,0.45), 0 1px 3px rgba(0,0,0,0.9)'
            : '0 1px 6px rgba(251,191,36,0.4), 0 1px 3px rgba(0,0,0,0.9)';
        const icon = L.divIcon({
          className: '',
          html: `<div class="officer-marker" style="border-color:${borderColor};box-shadow:${shadow}" title="${officer.name}">
                   <span style="color:${dotColor};margin-right:3px;">●</span>${officer.badge}
                 </div>`,
          iconSize: [68, 20],
          iconAnchor: [34, 10],
        });
        const marker = L.marker(oLatLng(officer), { icon, zIndexOffset: 100 });
        marker.on('click', () => onSelectOfficer(officer));
        marker.addTo(map);
        markersRef.current.push(marker);
      });

    return () => { markersRef.current.forEach(m => m.remove()); markersRef.current = []; };
  }, [officers, visible, map, onSelectOfficer]);

  return null;
}

// ── Main map component ────────────────────────────────────────────────────────
export default function MapView({
  junctions, officers, routes, layers,
  selectedJunction, activeRoute, incidentJunctionId,
  onSelectJunction, onSelectOfficer, onViewRoute,
}: Props) {
  // Default center: Nagpur city centre
  const center: [number, number] = [21.1458, 79.0882];

  return (
    <MapContainer
      center={center}
      zoom={13}
      style={{ width: '100%', height: '100%' }}
      zoomControl={true}
      attributionControl={true}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
        maxZoom={19}
        className="map-tile-brightened"
      />

      <MapController selectedJunction={selectedJunction} />
      <OfficerMarkers officers={officers} onSelectOfficer={onSelectOfficer} visible={layers.officers} />

      {/* White outline rings — separate pass so no fragment is needed */}
      {layers.risk && junctions.map(j => (
        <CircleMarker
          key={`ring-${j.id}`}
          center={jLatLng(j)}
          radius={riskRadius(j.riskScore) + 2}
          pathOptions={{
            color: 'rgba(255,255,255,0.55)',
            fillColor: 'transparent',
            fillOpacity: 0,
            weight: selectedJunction?.id === j.id ? 2 : 1.5,
            opacity: selectedJunction?.id === j.id ? 0.9 : 0.65,
          }}
          interactive={false}
        />
      ))}

      {/* Risk junction markers — driven entirely by dynamic junction data */}
      {layers.risk && junctions.map(j => {
        const isSelected   = selectedJunction?.id === j.id;
        const isIncident   = incidentJunctionId === j.id;
        const color        = RISK_COLORS[j.riskLevel];
        const isCritical   = j.riskLevel === 'CRITICAL';
        const isHigh       = j.riskLevel === 'HIGH';
        return (
          <CircleMarker
            key={j.id}
            center={jLatLng(j)}
            radius={riskRadius(j.riskScore)}
            pathOptions={{
              color,
              fillColor: color,
              fillOpacity: isSelected ? 0.55 : isCritical ? 0.38 : isHigh ? 0.32 : 0.22,
              weight: isSelected ? 2.5 : isCritical ? 2.5 : isHigh ? 2 : 1.5,
              opacity: 1,
            }}
            eventHandlers={{ click: () => onSelectJunction(j) }}
          >
            <Tooltip permanent={isSelected || isIncident} direction="top" offset={[0, -8]} className="">
              <div style={{
                background: '#0c1428', border: `1px solid ${color}40`, borderRadius: 3,
                padding: '2px 6px', fontSize: 10, color,
                fontFamily: 'JetBrains Mono, monospace', fontWeight: 600,
              }}>
                {j.id} · {j.riskScore}
              </div>
            </Tooltip>
            <Popup minWidth={220} maxWidth={280}>
              <JunctionPopup junction={j} onViewRoute={onViewRoute} />
            </Popup>
          </CircleMarker>
        );
      })}

      {/* Active Dijkstra route — bright cyan/blue with glow */}
      {layers.routes && activeRoute && (
        <>
          {/* Outer glow pass — wider, transparent */}
          <Polyline
            positions={activeRoute.path}
            pathOptions={{ color: '#93c5fd', weight: 9, opacity: 0.18, lineCap: 'round', lineJoin: 'round' }}
          />
          {/* Mid glow */}
          <Polyline
            positions={activeRoute.path}
            pathOptions={{ color: '#60a5fa', weight: 6, opacity: 0.32, lineCap: 'round', lineJoin: 'round' }}
          />
          {/* Core route line */}
          <Polyline
            className="route-glow-layer"
            positions={activeRoute.path}
            pathOptions={{ color: '#3b82f6', weight: 4, opacity: 0.95, lineCap: 'round', lineJoin: 'round' }}
          />
        </>
      )}

      {/* All allocation routes (dim background) */}
      {layers.routes && !activeRoute && routes.map(r => (
        <Polyline
          key={`${r.officerId}-${r.junctionId}`}
          positions={r.path}
          pathOptions={{ color: '#2563eb', weight: 2, opacity: 0.5, lineCap: 'round' }}
        />
      ))}
    </MapContainer>
  );
}

// ── Junction popup ────────────────────────────────────────────────────────────
function JunctionPopup({ junction: j, onViewRoute }: {
  junction: Junction;
  onViewRoute: (j: Junction) => void;
}) {
  const color = RISK_COLORS[j.riskLevel];
  return (
    <div style={{ padding: '12px 14px', minWidth: 210, fontFamily: 'Inter, sans-serif' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ color: '#c8d8f0', fontWeight: 700, fontSize: 12 }}>{j.id}</span>
        <span style={{
          background: `${color}22`, color, border: `1px solid ${color}60`,
          borderRadius: 2, padding: '1px 6px', fontSize: 10, fontWeight: 600,
        }}>{j.riskLevel}</span>
      </div>
      <div style={{ color: '#7a9abf', fontSize: 11, marginBottom: 10 }}>{j.name}</div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ color: '#4a6080', fontSize: 10 }}>Risk Score</span>
        <span style={{ color, fontFamily: 'JetBrains Mono, monospace', fontSize: 16, fontWeight: 700 }}>
          {j.riskScore}<span style={{ fontSize: 10, color: '#4a6080' }}>/100</span>
        </span>
      </div>

      <div style={{ borderTop: '1px solid #1c3050', paddingTop: 8, marginBottom: 10 }}>
        <div style={{ color: '#4a6080', fontSize: 10, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Risk Factors
        </div>
        {(Object.entries(j.factors) as [string, number][]).map(([key, val]) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ color: '#7a9abf', fontSize: 10, textTransform: 'capitalize' }}>
              {key.replace(/([A-Z])/g, ' $1')}
            </span>
            <span style={{ color: '#c8d8f0', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>{val}</span>
          </div>
        ))}
      </div>

      <div style={{ borderTop: '1px solid #1c3050', paddingTop: 8, marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#4a6080', fontSize: 10 }}>Congestion</span>
          <span style={{ color: '#ffd700', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            {j.congestionFactor.toFixed(2)}×
          </span>
        </div>
        {j.assignedOfficer && (
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#4a6080', fontSize: 10 }}>Assigned</span>
            <span style={{ color: '#c8d8f0', fontSize: 10 }}>{j.assignedOfficer}</span>
          </div>
        )}
      </div>

      <button
        onClick={() => onViewRoute(j)}
        style={{
          width: '100%', padding: '6px 0', background: '#1e3a6e', border: '1px solid #2563eb',
          color: '#93c5fd', borderRadius: 3, fontSize: 11, fontWeight: 500, cursor: 'pointer',
          fontFamily: 'Inter, sans-serif',
        }}
      >
        View Route
      </button>
    </div>
  );
}
