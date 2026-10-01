(() => {
  if (window.farmBoundaryExtensionInjected) return;
  window.farmBoundaryExtensionInjected = true;

  const MESSAGE_SOURCE = 'farmboundary-extension';
  const RESPONSE_SOURCE = 'farmboundary-extension-response';

  const TILE_SIZE = 256;

  function lonToPixel(lon, zoom) {
    const x = (lon + 180) / 360;
    const worldSize = TILE_SIZE * Math.pow(2, zoom);
    return x * worldSize;
  }

  function latToPixel(lat, zoom) {
    const sinLat = Math.sin((lat * Math.PI) / 180);
    const y = 0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI);
    const worldSize = TILE_SIZE * Math.pow(2, zoom);
    return y * worldSize;
  }

  function pixelToLon(x, zoom) {
    const worldSize = TILE_SIZE * Math.pow(2, zoom);
    const lon = (x / worldSize) * 360 - 180;
    return lon;
  }

  function pixelToLat(y, zoom) {
    const worldSize = TILE_SIZE * Math.pow(2, zoom);
    const yNorm = 0.5 - (y / worldSize);
    const lat = 90 - (360 * Math.atan(Math.exp(-yNorm * 2 * Math.PI))) / Math.PI;
    return lat;
  }
  
  if (window.location.hostname === "earth.google.com") {

    const earth = document.querySelector("earth-app");

    console.log("Earth mode");

  }

  function parseUrlCenterZoom(url) {
    const atMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z/);
    if (atMatch) {
      return {
        lat: parseFloat(atMatch[1]),
        lng: parseFloat(atMatch[2]),
        zoom: parseFloat(atMatch[3])
      };
    }

    const centerMatch = url.match(/[?&]center=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/);
    const zoomMatch = url.match(/[?&]zoom=(\d+)/);
    if (centerMatch && zoomMatch) {
      return {
        lat: parseFloat(centerMatch[1]),
        lng: parseFloat(centerMatch[2]),
        zoom: parseFloat(zoomMatch[1])
      };
    }

    const altMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(m|km)(?:[/?]|$)/);
    if (altMatch) {
      const meters = parseFloat(altMatch[3]) * (altMatch[4] === 'km' ? 1000 : 1);
      const estimatedZoom = Math.max(1, Math.min(21, 19 - Math.round(Math.log10(meters) - 2)));
      return {
        lat: parseFloat(altMatch[1]),
        lng: parseFloat(altMatch[2]),
        zoom: estimatedZoom
      };
    }

    const earthMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(m|km)a(?:,|$)/);
    if (earthMatch) {
      const meters = parseFloat(earthMatch[3]) * (earthMatch[4] === 'km' ? 1000 : 1);
      const estimatedZoom = Math.max(1, Math.min(21, 19 - Math.round(Math.log10(meters) - 2)));
      return {
        lat: parseFloat(earthMatch[1]),
        lng: parseFloat(earthMatch[2]),
        zoom: estimatedZoom
      };
    }

    const gmapMatch = url.match(/!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/);
    if (gmapMatch) {
      const zoomFallback = 18;
      return {
        lat: parseFloat(gmapMatch[1]),
        lng: parseFloat(gmapMatch[2]),
        zoom: zoomFallback
      };
    }

    const hashMatch = url.match(/#@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z/);
    if (hashMatch) {
      return {
        lat: parseFloat(hashMatch[1]),
        lng: parseFloat(hashMatch[2]),
        zoom: parseFloat(hashMatch[3])
      };
    }

    return null;
  }

  function getMapContainer() {
    return document.querySelector('div[role="application"]') || document.body;
  }

  function getViewportSize() {
    const container = getMapContainer();
    return {
      width: container.clientWidth || window.innerWidth,
      height: container.clientHeight || window.innerHeight
    };
  }

  function getMapStateFromUrl() {
    const url = window.location.href;
    const centerZoom = parseUrlCenterZoom(url);
    const viewport = getViewportSize();

    if (!centerZoom) {
      return {
        url,
        success: false,
        message: 'Unable to parse map center/zoom from URL.'
      };
    }

    const { lat, lng, zoom } = centerZoom;
    const centerX = lonToPixel(lng, zoom);
    const centerY = latToPixel(lat, zoom);
    const halfWidth = viewport.width / 2;
    const halfHeight = viewport.height / 2;

    const nwX = centerX - halfWidth;
    const nwY = centerY - halfHeight;
    const seX = centerX + halfWidth;
    const seY = centerY + halfHeight;

    return {
      url,
      success: true,
      center: { lat, lng },
      zoom,
      bounds: {
        north: pixelToLat(nwY, zoom),
        west: pixelToLon(nwX, zoom),
        south: pixelToLat(seY, zoom),
        east: pixelToLon(seX, zoom)
      },
      width: viewport.width,
      height: viewport.height,
      pageTitle: document.title
    };
  }

  function createOverlay() {
    const existing = document.getElementById('farmboundary-overlay-container');
    if (existing) return existing;

    const container = document.createElement('div');
    container.id = 'farmboundary-overlay-container';
    container.style.position = 'fixed';
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.pointerEvents = 'none';
    container.style.zIndex = '999999999';
    container.style.mixBlendMode = 'normal';

    const canvas = document.createElement('canvas');
    canvas.id = 'farmboundary-overlay-canvas';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    container.appendChild(canvas);
    document.body.appendChild(container);
    return container;
  }

  function clearOverlay() {
    const container = document.getElementById('farmboundary-overlay-container');
    if (container) {
      container.remove();
    }
  }

  function drawGeoJSON(geojson) {
    const overlay = createOverlay();
    const canvas = overlay.querySelector('#farmboundary-overlay-canvas');
    const rect = overlay.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const state = getMapStateFromUrl();
    if (!state.success) {
      return false;
    }

    const bounds = state.bounds;
    const width = state.width;
    const height = state.height;

    function projectPoint(lng, lat) {
      const x = ((lng - bounds.west) / (bounds.east - bounds.west)) * width;
      const y = ((bounds.north - lat) / (bounds.north - bounds.south)) * height;
      return { x, y };
    }

    geojson.features.forEach(feature => {
      const coords = feature.geometry.coordinates[0].map(([lng, lat]) => projectPoint(lng, lat));
      if (coords.length < 2) return;

      ctx.beginPath();
      coords.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.18)';
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();
    });

    return true;
  }

  window.addEventListener('message', event => {
    if (!event.data || event.source !== window) return;
    if (event.data.source !== MESSAGE_SOURCE) return;

    const { type, id, payload } = event.data;
    let response = { id, source: RESPONSE_SOURCE };

    try {
      if (type === 'get-map-state') {
        response.payload = getMapStateFromUrl();
      } else if (type === 'draw-geojson') {
        response.payload = { success: drawGeoJSON(payload.geojson) };
      } else if (type === 'clear-overlay') {
        clearOverlay();
        response.payload = { success: true };
      } else {
        response.payload = { success: false, message: 'Unknown command' };
      }
    } catch (err) {
      response.payload = { success: false, message: err.message };
    }

    window.postMessage(response, '*');
  });
})();
