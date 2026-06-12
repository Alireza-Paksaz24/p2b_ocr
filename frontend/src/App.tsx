import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { lazy, Suspense } from 'react';

// Lazy-loaded pages
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Models = lazy(() => import('./pages/Models'));
const NewJob = lazy(() => import('./pages/NewJob'));
const Jobs = lazy(() => import('./pages/Jobs'));
const JobDetail = lazy(() => import('./pages/JobDetail'));
const History = lazy(() => import('./pages/History'));

function PageFallback() {
  return <div className="flex items-center justify-center h-full">Loading...</div>;
}

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="models" element={<Models />} />
            <Route path="new-job" element={<NewJob />} />
            <Route path="jobs" element={<Jobs />} />
            <Route path="jobs/:jobId" element={<JobDetail />} />
            <Route path="history" element={<History />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}