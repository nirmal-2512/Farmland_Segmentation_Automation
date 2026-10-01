function getVisibleMapBounds() {
  if (!window.google || !window.google.maps) {
    return null;
  }

  const maps = document.querySelector('div[role="application"]');
  if (!maps) {
    return null;
  }

  const map = window.map || null;
  if (!map) {
    return null;
  }

  const bounds = map.getBounds();
  const center = map.getCenter();
  const zoom = map.getZoom();

  return {
    north: bounds.getNorthEast().lat(),
    east: bounds.getNorthEast().lng(),
    south: bounds.getSouthWest().lat(),
    west: bounds.getSouthWest().lng(),
    center: { lat: center.lat(), lng: center.lng() },
    zoom: zoom
  };
}

function getVisibleMapSize() {
  const mapContainer = document.querySelector('div[role="application"]');
  if (!mapContainer) {
    return null;
  }

  return {
    width: mapContainer.offsetWidth,
    height: mapContainer.offsetHeight
  };
}

export { getVisibleMapBounds, getVisibleMapSize };
