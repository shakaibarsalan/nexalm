import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const container = document.getElementById('root');
if (!container) throw new Error('root element missing');
const root = createRoot(container);

async function boot() {
  // `vite build --mode demo` swaps the server for an in-browser fake. In every
  // other build this branch is dropped, demo code and all.
  if (import.meta.env.MODE === 'demo') {
    const [{ installDemo }, { DemoFrame }] = await Promise.all([
      import('./demo/demoServer'),
      import('./demo/DemoBanner'),
    ]);
    installDemo();
    root.render(
      <StrictMode>
        <DemoFrame>
          <App />
        </DemoFrame>
      </StrictMode>,
    );
    return;
  }

  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void boot();
