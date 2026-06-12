import { NavLink, Outlet } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/models', label: 'Models' },
  { to: '/new-job', label: 'New Job' },
  { to: '/jobs', label: 'Jobs' },
  { to: '/history', label: 'History' },
];

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-800 text-white flex-shrink-0">
        <div className="p-4 text-xl font-bold border-b border-gray-700">OCR Service</div>
        <nav className="mt-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `block px-4 py-2 text-sm hover:bg-gray-700 ${isActive ? 'bg-gray-700 font-medium' : ''}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}