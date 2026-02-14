// ============================================================
// TCE ProjectForty2 — Shared NavBar Component
// ============================================================
// Load via: <script type="text/babel" src="shared-nav.js"></script>
// Requires React (useState from window scope) and .glass CSS class.

function NavBar() {
  const path = window.location.pathname;
  const links = [
    { label: 'Lab', href: '/pf2/lab.html' },
    { label: 'Chat', href: '/pf2/chat.html' },
    { label: 'Benchmarks', href: '/pf2/benchmarks.html' },
    { label: 'About', href: '/pf2/' },
  ];
  return (
    <nav className="sticky top-0 z-50 px-6 py-3 flex items-center justify-between border-b border-white/5 glass" aria-label="Main navigation">
      <a href="/pf2/" style={{ textDecoration: 'none' }} className="flex items-center gap-2">
        <span className="text-lg font-bold font-mono" style={{ background: 'linear-gradient(to right, #22d3ee, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>ProjectForty2</span>
      </a>
      <div className="flex items-center gap-1">
        {links.map(l => {
          const active = path === l.href || (l.href === '/pf2/' && (path === '/pf2' || path === '/pf2/' || path === '/pf2/index.html'))
            || (l.label === 'Lab' && path === '/lab.html') || (l.label === 'Chat' && path === '/chat.html') || (l.label === 'Benchmarks' && path === '/benchmarks.html');
          return (
            <a key={l.href} href={l.href} style={{ textDecoration: 'none' }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${active ? 'bg-white/10 text-white' : 'text-white/40 hover:text-white/70 hover:bg-white/5'}`}>
              {l.label}
            </a>
          );
        })}
      </div>
    </nav>
  );
}
