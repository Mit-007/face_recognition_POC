import axios from "axios";

const API_BASE = "http://localhost:8000";

export const registerPerson = (firstName, lastName, file) => {
  const form = new FormData();
  form.append("file", file);

  const query = `first_name=${encodeURIComponent(firstName)}&last_name=${encodeURIComponent(lastName || "")}`;
  return axios.post(`${API_BASE}/register_new_person?${query}`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const detectPerson = (file) => {
  const form = new FormData();
  form.append("file", file);

  return axios.post(`${API_BASE}/detect_person`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const imageUrl = (imageName) => `${API_BASE}/images/${imageName}`;

export const getDetectionHistory = (limit = 100) => {
  return axios.get(`${API_BASE}/detection_history?limit=${limit}`);
};

export const clearDetectionHistory = () => {
  return axios.delete(`${API_BASE}/detection_history`);
};

export const getRegisteredUsers = () => {
  return axios.get(`${API_BASE}/registered_users`);
};

export const deleteRegisteredUser = (personId) => {
  return axios.delete(`${API_BASE}/registered_users/${personId}`);
};
