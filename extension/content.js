if (window.farmBoundaryContentScriptInjected) {
  console.log('Farm boundary content script already injected.');
  return;
}
window.farmBoundaryContentScriptInjected = true;

const MESSAGE_SOURCE = 'farmboundary-extension';
const RESPONSE_SOURCE = 'farmboundary-extension-response';

function injectPageScript() {
  const existing = document.getElementById('farmboundary-injected-script');
  if (existing) return;

  const script = document.createElement('script');
  script.id = 'farmboundary-injected-script';
  script.src = chrome.runtime.getURL('injected.js');
  script.onload = () => script.remove();
  (document.documentElement || document.head || document.body).appendChild(script);
}

function waitForMapState(timeoutMs = 3000) {
  return new Promise((resolve) => {
    const messageId = `map-state-${Date.now()}`;
    const listener = event => {
      if (!event.data || event.data.source !== RESPONSE_SOURCE) return;
      if (event.data.id !== messageId) return;
      window.removeEventListener('message', listener);
      resolve(event.data.payload);
    };

    window.addEventListener('message', listener);
    window.postMessage({ source: MESSAGE_SOURCE, type: 'get-map-state', id: messageId }, '*');

    setTimeout(() => {
      window.removeEventListener('message', listener);
      resolve({ success: false, message: 'Map state timeout' });
    }, timeoutMs);
  });
}

function sendDrawCommand(geojson) {
  const messageId = `draw-geojson-${Date.now()}`;
  return new Promise((resolve) => {
    const listener = event => {
      if (!event.data || event.data.source !== RESPONSE_SOURCE) return;
      if (event.data.id !== messageId) return;
      window.removeEventListener('message', listener);
      resolve(event.data.payload);
    };

    window.addEventListener('message', listener);
    window.postMessage({ source: MESSAGE_SOURCE, type: 'draw-geojson', id: messageId, payload: { geojson } }, '*');
  });
}

function sendClearOverlay() {
  const messageId = `clear-overlay-${Date.now()}`;
  return new Promise((resolve) => {
    const listener = event => {
      if (!event.data || event.data.source !== RESPONSE_SOURCE) return;
      if (event.data.id !== messageId) return;
      window.removeEventListener('message', listener);
      resolve(event.data.payload);
    };

    window.addEventListener('message', listener);
    window.postMessage({ source: MESSAGE_SOURCE, type: 'clear-overlay', id: messageId }, '*');
  });
}

injectPageScript();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'get-map-state') {
    waitForMapState(3000).then(payload => sendResponse(payload));
    return true;
  }

  if (message && message.type === 'draw-geojson') {
    sendDrawCommand(message.geojson).then(payload => sendResponse(payload));
    return true;
  }

  if (message && message.type === 'clear-overlay') {
    sendClearOverlay().then(payload => sendResponse(payload));
    return true;
  }
});
