export default function Navbar({ tab, setTab }) {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <span className="brand-mark" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="21" r="7" stroke="#fff" strokeWidth="2.6" />
            <path
              d="M24 30c-6 0-10 3-10 6.5V38h20v-1.5c0-3.5-4-6.5-10-6.5Z"
              fill="#fff"
            />
          </svg>
        </span>
        <div>
          <p className="brand-name">FaceID</p>
          <p className="brand-tagline">Recognition System</p>
        </div>
      </div>

      <nav className="navbar-tabs" aria-label="Primary navigation">
        <button
          type="button"
          className={tab === "register" ? "active" : ""}
          aria-current={tab === "register" ? "page" : undefined}
          onClick={() => setTab("register")}
        >
          Register Person
        </button>
        <button
          type="button"
          className={tab === "detect" ? "active" : ""}
          aria-current={tab === "detect" ? "page" : undefined}
          onClick={() => setTab("detect")}
        >
          Detect Person
        </button>
        <button
          type="button"
          className={tab === "users" ? "active" : ""}
          aria-current={tab === "users" ? "page" : undefined}
          onClick={() => setTab("users")}
        >
          Registered Users
        </button>
        <button
          type="button"
          className={tab === "history" ? "active" : ""}
          aria-current={tab === "history" ? "page" : undefined}
          onClick={() => setTab("history")}
        >
          Detection History
        </button>
      </nav>
    </header>
  );
}
