import { useState } from "react";
import Navbar from "./components/Navbar";
import RegisterPerson from "./components/RegisterPerson";
import DetectPerson from "./components/DetectPerson";
import "./App.css";

function App() {
  const [tab, setTab] = useState("register");

  return (
    <div className="page">
      <Navbar tab={tab} setTab={setTab} />

      <main className="app">
        {tab === "register" ? <RegisterPerson /> : <DetectPerson />}
      </main>

      <footer className="app-footer">
        <p>FaceID Recognition System &middot; Internal POC</p>
      </footer>
    </div>
  );
}

export default App;
