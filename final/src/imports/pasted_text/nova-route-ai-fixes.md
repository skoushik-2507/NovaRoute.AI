Perform a FINAL CODE-QUALITY AND FUNCTIONALITY FIX on the existing NovaRoute.AI frontend.

IMPORTANT:
The current UI/design is APPROVED and must be preserved.

DO NOT redesign anything.
DO NOT change the visual theme.
DO NOT change colors, typography, spacing, cards, map styling, sidebar layout, filter layout, or navigation appearance.
DO NOT add new features.
DO NOT add unnecessary dependencies.
DO NOT rewrite working components unnecessarily.

Your job is ONLY to fix the important functional and architectural issues listed below.

==================================================
1. FIX REFRESH FUNCTIONALITY
==================================================

The current Refresh implementation can refresh data and then call reloadData(), which can overwrite the newly refreshed mock data.

Fix this so there is ONE clean refresh flow:

Refresh button
→ refresh current data through the existing API/service layer
→ update all relevant React state
→ update dashboard/map
→ finish

Do NOT refresh twice.

When Demo/Mock mode is active:
- regenerate/reload the mock dataset once
- update junctions
- officers
- traffic
- allocations
- routes
- KPIs
- map

When real backend mode is active:
- request the latest data through src/services/api.ts

During refresh:
- show loading state on the existing Refresh button
- disable it temporarily
- prevent duplicate refresh requests
- preserve the current sidebar page
- preserve selected filters
- preserve selected officer/junction where possible

Do not change the appearance of the Refresh button.

==================================================
2. FIX DEMO MODE VS REAL BACKEND MODE
==================================================

Make Demo Mode behavior explicit and reliable.

There must be a clear distinction between:

DEMO ON:
- use mock/simulated data
- existing demo functionality continues working

DEMO OFF:
- use the real API/backend
- do not silently pretend the backend is live when mock fallback is being used

Continue using:

VITE_API_URL
VITE_USE_MOCK

with the existing API service architecture.

Default behavior may remain mock/demo if VITE_USE_MOCK is not configured, so the hackathon demo does not break.

However, when VITE_USE_MOCK=false:
- attempt the real backend
- do not silently report "SYSTEM LIVE" if the backend request failed

==================================================
3. FIX BACKEND STATUS / FALLBACK HANDLING
==================================================

The current API fallback can catch backend errors and return mock data.

Preserve the fallback because it is useful for the hackathon.

BUT distinguish these states:

BACKEND CONNECTED:
- real backend data is being used

MOCK/DEMO:
- mock data is intentionally being used

BACKEND UNAVAILABLE / FALLBACK:
- backend was attempted but failed
- mock fallback is being used

Do NOT show "SYSTEM LIVE" when the backend is actually unavailable and mock fallback data is being displayed.

Expose a simple connection/status state to the UI using the existing styling.

Do not redesign the header.

==================================================
4. FIX TRAFFIC DATA TYPE CONSISTENCY
==================================================

Inspect src/types/api.ts and all traffic-related code.

TrafficData currently has a timestamp-related inconsistency.

Choose ONE canonical field and use it consistently.

Prefer:

timestamp: string

Remove unnecessary duplicate fields such as lastUpdated if timestamp already provides the required information.

Update:
- types
- mock data
- API responses
- hooks
- UI references

so TypeScript has no mismatch.

Do not use `as any` to hide the problem.

==================================================
5. REDUCE IMPORTANT `any` USAGE
==================================================

Do NOT attempt a huge refactor.

Only replace `any` where it affects important application data.

At minimum, use the existing types for:

- TrafficData
- CoverageData
- BaselineData
- OfficerAllocation
- Incident
- DeploymentResult
- Junction
- Officer
- Route

Use the existing src/types/api.ts definitions.

Do not invent duplicate types if an appropriate type already exists.

Do not use `any` simply to suppress TypeScript errors.

==================================================
6. FIX INCIDENT SIMULATION STATE FLOW
==================================================

Preserve the current Incident Simulation UI and behavior.

When an incident is simulated, the frontend should update the relevant application state consistently.

The resulting deployment/reallocation response should be capable of updating:

- affected junction
- risk
- traffic/congestion
- officer allocation
- routes
- officer status
- response time
- coverage where applicable

For now, mock mode may use simulated data.

For real backend mode, prepare the state flow so the backend DeploymentResult can update the corresponding frontend state.

Do NOT invent fake backend endpoints.

Continue using src/services/api.ts.

==================================================
7. FIX ROUTE / ALLOCATION API CONTRACT
==================================================

Inspect getAllRoutes() and the existing API service.

Do not assume that `/api/allocations` is automatically the route endpoint unless the existing project explicitly requires it.

Keep the endpoint centralized in src/services/api.ts.

If the backend contract is not yet known:
- keep the endpoint configurable/centralized
- do not scatter route URLs throughout components
- do not invent a new backend implementation

The frontend Route type must continue supporting:

officerId
junctionId
responseTime
distance
path

where path is suitable for Leaflet rendering.

==================================================
8. FIX TIME FILTER LOGIC
==================================================

The Time filter must not be purely decorative.

Keep the existing Time dropdown and UI unchanged.

Ensure selectedTime is part of the filtering/data logic.

Support:

Live
Last 15 min
Last 30 min
Last 1 hour
Today

For mock data:
- filter where timestamps support it
- if the current mock dataset does not contain sufficient historical timestamps, do NOT fabricate unrealistic historical data

For real backend data:
- keep the filter state ready to be passed to/used by the API layer later.

Do not break the existing Risk, Traffic, Officers, or Area filters.

==================================================
9. PRESERVE AND VERIFY FILTER FUNCTIONALITY
==================================================

Do not redesign the filters.

Verify that:

Risk
Traffic
Officers
Area
Time

all:
- open correctly
- select values
- close correctly
- update displayed data where supported
- work together
- preserve their selected state during Refresh
- continue working after Incident Simulation

Risk filter should affect risk markers/ranking.

Traffic filter should affect traffic-related data/layers where supported.

Officer filter should affect officer list/markers.

Area filter should affect relevant data.

Time filter should participate in filtering where timestamp data exists.

Reset must continue working.

==================================================
10. PRESERVE SIDEBAR NAVIGATION
==================================================

Do not redesign the sidebar.

Verify that every existing navigation item works:

Dashboard
Risk Intelligence
Live Traffic
Officers
Deployment
Incidents
Coverage
Performance

Clicking an item must:
- change the displayed view
- update the active navigation state
- not reload the browser
- preserve the existing styling

Do not create unnecessary duplicate implementations.

==================================================
11. PRESERVE MAP FUNCTIONALITY
==================================================

Do not redesign the map.

Verify:

- risk markers work
- officer markers work
- route visualization works
- traffic layer works
- incident layer works
- layer toggles work
- filters affect appropriate layers
- routes remain visible
- selected junction/officer interactions still work

The existing route visualization should remain compatible with backend-generated Dijkstra paths.

==================================================
12. REMOVE ONLY UNUSED / UNNECESSARY PROJECT CONTENT
==================================================

Inspect:

src/imports/pasted_text/novaroute-ai-dashboard.tsx

If it is genuinely unused by the application and only contains the original generation prompt/pasted content, remove it from the production source tree.

Do NOT remove anything that is actually imported or required.

==================================================
13. DO NOT REFACTOR App.tsx AGGRESSIVELY
==================================================

The current App.tsx is large.

DO NOT perform a massive component refactor now.

Do not risk breaking the working hackathon UI.

Only make targeted changes required for the fixes above.

==================================================
14. PRESERVE MOCK DATA
==================================================

Do not remove mock/demo data.

The frontend must remain usable if the backend is unavailable.

Mock data should continue to support:

- dashboard
- risk
- traffic
- officers
- deployment
- incidents
- coverage
- performance
- map
- routes

==================================================
15. FINAL VALIDATION
==================================================

After making the fixes:

Check all imports.

Check TypeScript types.

Check API service.

Check mock fallback.

Check Demo Mode.

Check Refresh.

Check filters.

Check sidebar navigation.

Check map.

Check routes.

Check officer locking.

Check Incident Simulation.

Check Deployment.

Check Coverage.

Check Performance.

Make sure there are:

ZERO unresolved imports
ZERO runtime errors
ZERO obvious TypeScript errors
ZERO broken navigation items
ZERO broken filter controls

Run the project build.

If the build reports an error, fix the actual root cause.

Do NOT hide errors with:
- @ts-ignore
- eslint-disable
- any
- empty catch blocks
- error suppression

==================================================
FINAL RULE
==================================================

The UI is already approved.

Do not redesign it.

Do not add features.

Do not change the architecture unnecessarily.

Make only the important fixes above and leave the application visually and functionally stable.

The final result must be:

NovaRoute.AI frontend
+ approved UI
+ working navigation
+ working filters
+ working Refresh
+ reliable Demo/Mock mode
+ clear backend/fallback status
+ clean API service layer
+ consistent TypeScript types
+ backend-integration ready
+ ZERO build/runtime errors.