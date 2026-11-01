import axios from "axios";

const API_BASE = "http://localhost:8000";

export const registerPerson = (name, file) => {
  const form = new FormData();
  form.append("file", file);

  return axios.post(
    `${API_BASE}/register_new_person?name=${encodeURIComponent(name)}`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
};

export const detectPerson = (file) => {
  const form = new FormData();
  form.append("file", file);

  return axios.post(`${API_BASE}/detect_person`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const imageUrl = (imageName) => `${API_BASE}/images/${imageName}`;
