let mapPolygons = [];

function clearMapPolygons() {
  mapPolygons.forEach(poly => poly.setMap(null));
  mapPolygons = [];
}

function drawGeoJSONOnMap(geojson) {
  clearMapPolygons();

  if (!window.google || !window.google.maps) {
    console.error('Google Maps API not available.');
    return;
  }

  const map = window.map || window.google.maps.Map ? window.map : null;

  if (!map) {
    console.warn('Could not find map instance.');
  }

  geojson.features.forEach(feature => {
    const coords = feature.geometry.coordinates[0].map(([lng, lat]) => ({ lat, lng }));
    const polygon = new google.maps.Polygon({
      paths: coords,
      strokeColor: '#FF0000',
      strokeOpacity: 0.8,
      strokeWeight: 2,
      fillColor: '#FF0000',
      fillOpacity: 0.25
    });
    polygon.setMap(map);
    mapPolygons.push(polygon);
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'draw-geojson') {
    drawGeoJSONOnMap(message.geojson);
    sendResponse({ status: 'drawn' });
    return true;
  }
});
