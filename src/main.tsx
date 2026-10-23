import React from "react";
import ReactDOM from "react-dom/client";
import { ReactFlowProvider } from "@xyflow/react";
import App from "./App";
import AppErrorBoundary from "./components/AppErrorBoundary";
import "@xyflow/react/dist/style.css";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <ReactFlowProvider>
        <App />
      </ReactFlowProvider>
    </AppErrorBoundary>
  </React.StrictMode>,
);
