import { useEffect, useState } from "react";
import { getRegisteredUsers, deleteRegisteredUser, imageUrl } from "../api";

export default function RegisteredUsers({ onNavigateRegister }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await getRegisteredUsers();
      setUsers(res.data.users || []);
    } catch {
      // Fallback on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    getRegisteredUsers()
      .then((res) => {
        if (!ignore) {
          setUsers(res.data.users || []);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, []);

  const handleDeleteUser = async (personId, personName) => {
    if (!window.confirm(`Are you sure you want to remove "${personName}" from the database?`)) {
      return;
    }
    try {
      await deleteRegisteredUser(personId);
      setUsers((prev) => prev.filter((u) => u.person_id !== personId));
    } catch {
      alert("Failed to delete user.");
    }
  };

  const filteredUsers = users.filter((u) => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return true;
    return (
      (u.name && u.name.toLowerCase().includes(q)) ||
      (u.registered_at && u.registered_at.includes(q))
    );
  });

  return (
    <div className="card registered-users-card">
      <div className="users-section-header">
        <div>
          <h2>Registered Users Database</h2>
          <p className="subtitle">
            All enrolled identities stored in <code>registered_users.json</code> and Pinecone.
          </p>
        </div>
        <div className="users-header-actions">
          <input
            type="text"
            placeholder="Search by name..."
            className="users-search-input"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button
            type="button"
            className="history-btn secondary"
            onClick={fetchUsers}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "↻ Refresh"}
          </button>
          {onNavigateRegister && (
            <button
              type="button"
              className="camera-btn"
              onClick={onNavigateRegister}
              style={{ padding: "7px 14px", fontSize: "12.5px" }}
            >
              + Register New
            </button>
          )}
        </div>
      </div>

      {loading && users.length === 0 ? (
        <div className="users-loading">
          <span className="spinner dark" />
          <p>Loading registered identities from database...</p>
        </div>
      ) : filteredUsers.length === 0 ? (
        <div className="users-empty">
          <span className="empty-icon">👥</span>
          <h4>No registered identities found</h4>
          <p>
            {users.length === 0
              ? "No users have been registered yet. Navigate to 'Register Person' to enroll identities."
              : "No registered persons match your search filter."}
          </p>
          {users.length === 0 && onNavigateRegister && (
            <button
              type="button"
              className="camera-btn"
              onClick={onNavigateRegister}
              style={{ marginTop: "8px" }}
            >
              Enroll First Person
            </button>
          )}
        </div>
      ) : (
        <div className="registered-users-grid">
          {filteredUsers.map((u) => (
            <div key={u.person_id} className="user-profile-card">
              <img
                className="user-card-avatar"
                src={imageUrl(u.image_name)}
                alt={u.name}
              />
              <div className="user-card-body">
                <strong className="user-card-name">{u.name}</strong>
                <span className="user-card-date">
                  {u.registered_at ? `Enrolled: ${u.registered_at}` : "Verified Identity"}
                </span>
              </div>
              <button
                type="button"
                className="user-delete-btn"
                title="Remove from database"
                onClick={() => handleDeleteUser(u.person_id, u.name)}
              >
                ✕ Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
