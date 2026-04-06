import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

console.log('index.tsx script started.');

const container = document.getElementById('root');

if (!container) {
  console.error('Root element #root not found!');
} else {
  console.log('Root element #root found.', container);
  const root = ReactDOM.createRoot(container);
  console.log('React root created.', root);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  console.log('App rendered.');
} 