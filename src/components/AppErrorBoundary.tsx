import { Component, type ReactNode } from "react";

export default class AppErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <main className="app-recovery" role="alert">
          <h1>暂时无法显示此模型</h1>
          <p>配置或场景超出当前支持范围。重新加载可恢复默认模型。</p>
          <button
            onClick={() => window.location.assign(window.location.pathname)}
          >
            恢复默认模型
          </button>
        </main>
      );
    return this.props.children;
  }
}
