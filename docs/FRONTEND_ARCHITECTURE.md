# UI Architecture: CrowdDNA

## Philosophy
The CrowdDNA Frontend operates as a completely decoupled consumer of the Phase 22 API Layer. It implements a strict **Feature-Based Architecture**, ensuring that complex domain concepts (e.g., Explainability, Reproducibility) do not bleed into global state or pollute unrelated components.

## Technical Stack
- **Framework**: React.js (via Vite)
- **Language**: TypeScript (for strict API DTO mapping, generated via OpenAPI)
- **Server State Management**: TanStack Query (Caching, invalidation, retries)
- **UI State Management**: Zustand (Immutable UI state, modals, sidebars)
- **Styling**: Vanilla CSS Modules (Variables, Glassmorphism, Micro-animations)
- **Routing**: React Router DOM (with route-level RBAC guards and Feature Flags)
- **Data Fetching**: Native Fetch API wrapped in modular clients

## Directory Structure

```
frontend/
├── src/
│   ├── config/       # Frontend Configuration
│   │   ├── env.ts
│   │   ├── constants.ts
│   │   └── featureFlags.ts (Route-level feature toggling)
│   ├── api/          # Modular Phase 22 API clients (OpenAPI Generated)
│   │   ├── client.ts
│   │   ├── interceptors.ts
│   │   ├── retry.ts
│   │   ├── errors.ts
│   │   └── streaming.ts
│   ├── assets/       # Centralized Assets Package
│   │   ├── icons/
│   │   ├── logos/
│   │   ├── illustrations/
│   │   └── fonts/
│   ├── providers/    # Global React Providers
│   │   ├── AuthProvider.tsx
│   │   ├── QueryProvider.tsx
│   │   ├── ThemeProvider.tsx
│   │   ├── NotificationProvider.tsx
│   │   ├── RealtimeProvider.tsx
│   │   └── ModalProvider.tsx
│   ├── auth/         # JWT management and token refresh
│   ├── components/   # Reusable Design System
│   │   ├── ui/       # Core primitives (Buttons, Inputs)
│   │   ├── layout/   # Grid, Flex, Containers
│   │   └── feedback/ # Toasts, Alerts, Progress
│   ├── layouts/      # Application shells (ErrorBoundary, Suspense, Sidebar)
│   ├── routing/      # Global router and role-based guards
│   │   └── lazy.ts   # Route-level code splitting via React.lazy
│   ├── stores/       # Global Zustand stores (UI state exclusively)
│   ├── theme/        # CSS variables, typography, and animation definitions
│   │   ├── tokens.css
│   │   ├── variables.css
│   │   ├── typography.css
│   │   ├── spacing.css
│   │   ├── animations.css
│   │   └── glass.css
│   ├── types/        # Global TypeScript interfaces mapped to Backend DTOs
│   ├── utils/        # Shared helper functions (Upload subsystem)
│   ├── features/     # Feature-sliced modules
│   │   ├── explainability/
│   │   │   ├── api/
│   │   │   ├── components/
│   │   │   ├── pages/
│   │   │   └── hooks/
│   │   ├── inference/
│   │   ├── observability/
│   │   ├── ...
```

## Core Systems

### 1. Configuration & Assets
The frontend is strictly parameterized via `config/env.ts` and `config/featureFlags.ts`. Routes and major UI branches can be dynamically toggled via feature flags. Static assets (fonts, icons, illustrations) are bundled centrally in `assets/` to guarantee deterministic build outputs and optimized loading.

### 2. OpenAPI Generated Client & TanStack Query
The Phase 22 OpenAPI specification is the canonical frontend contract. DTOs and API client types are strictly generated from the OpenAPI schema, absolutely prohibiting manual duplication of backend data types.
The API layer is heavily modularized to maintain strict architectural boundaries:
- **Interceptors**: Attaches Identity tokens automatically and intercepts token expiration (401) for seamless background refresh.
- **Errors**: Funnels 400/500 errors to the `NotificationProvider`.
- **Retry Logic**: Safely queues offline requests, ensuring retries strictly apply only to idempotent requests or those possessing an `Idempotency-Key` header.
- **TanStack Query**: Abstracts caching, staleness, and background synchronization entirely out of the UI layer.

### 3. Realtime Client & Provider Abstraction
The streaming architecture is split in two:
- **RealtimeClient**: A pure TypeScript class handling the transport mechanics (SSE, WebSocket, auto-reconnect, fallback to polling).
- **RealtimeProvider**: A React Context bridging the client into the React lifecycle and component tree.

### 4. Reusable Design System
All foundational UI elements exist strictly inside `components/ui/`, `components/layout/`, and `components/feedback/`. Feature modules must compose these primitive blocks rather than implementing their own raw styling.

### 5. Upload & File Management Subsystem
Dedicated utilities manage file transmission (e.g. video uploads for inference), offering resumability via chunking and deterministic progress tracking through the UI. Interrupted uploads automatically recover across page refreshes.

### 6. Robust Component Boundaries & Lazy Loading
- **ErrorBoundary**: Top-level and route-level boundaries catch rendering faults without white-screening the SPA.
- **Suspense & Code Splitting**: Every feature module is lazy-loaded via `React.lazy` inside `routing/lazy.ts`, coordinated by `Suspense` boundaries for minimal initial load sizes.
- **Virtualization**: `react-virtual` or similar minimalist abstractions are used for observability logs, avoiding DOM explosions.

### 7. Role-Based Access Control (RBAC)
The UI never computes permissions. The backend provides a `permissions` array in the login DTO. The frontend uses `ProtectedRoute` wrappers to redirect unauthorized users. Permissions sync dynamically in the background without requiring a hard refresh.

### 8. Accessibility & Aesthetics
The UI guarantees strict A11y standards:
- **Keyboard Navigation**: Fully tab-traversable forms and tables.
- **ARIA Management**: Precise labeling and dynamic live regions for notifications.
- **Focus Management**: Controlled focus retention inside modals.
- **Reduced-Motion**: Respects OS-level `prefers-reduced-motion` to tone down transitions.
- **Modular CSS Tokens**: Standardizes a glassmorphic interface spanning colors, typography, spacing, and micro-animations via `theme/*.css`. Supports dynamic toggling of dark mode.
