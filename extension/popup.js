const captureButton =
  document.getElementById(
    'captureButton'
  );

const automationButton =
  document.getElementById(
    'automationButton'
  );

const clearOverlayButton =
  document.getElementById(
    'clearOverlayButton'
  );

const downloadButton =
  document.getElementById(
    'downloadButton'
  );

const downloadKmlButton =
  document.getElementById(
    'downloadKmlButton'
  );

const statusText =
  document.getElementById(
    'status'
  );

const geojsonText =
  document.getElementById(
    'geojsonText'
  );


let latestGeoJSON = null;


/*
 * ---------------------------------------------------------
 * FastAPI
 * ---------------------------------------------------------
 */

const FASTAPI_URL =
  'http://localhost:8001';


/*
 * ---------------------------------------------------------
 * Status
 * ---------------------------------------------------------
 */

function updateStatus(
  message
) {

  statusText.innerHTML =
    message;
}


/*
 * ---------------------------------------------------------
 * Automation status listener
 *
 * Background service worker sends status here.
 * ---------------------------------------------------------
 */

chrome.runtime.onMessage.addListener(
  message => {

    if (
      !message ||
      message.type !==
        'automation-status'
    ) {
      return;
    }


    console.log(
      '[Automation]',
      message.status,
      message.message
    );


    updateStatus(
      message.message
    );


    if (
      message.status ===
      'starting'
    ) {

      automationButton.disabled =
        true;

      captureButton.disabled =
        true;
    }


    if (
      message.status ===
        'completed' ||
      message.status ===
        'error'
    ) {

      automationButton.disabled =
        false;

      captureButton.disabled =
        false;
    }
  }
);


/*
 * ---------------------------------------------------------
 * Start automation
 * ---------------------------------------------------------
 */

async function startAutomation() {

  try {

    automationButton.disabled =
      true;

    captureButton.disabled =
      true;


    updateStatus(
      'Starting tile automation...'
    );


    const response =
      await chrome.runtime.sendMessage({
        type:
          'start-single-tile-automation'
      });


    if (
      !response ||
      !response.success
    ) {

      throw new Error(
        response?.message ||
        'Unable to start automation.'
      );
    }


    /*
     * Do NOT wait here.
     *
     * The background service worker
     * continues the automation even
     * if the popup closes.
     */

    updateStatus(
      'Automation running in background...'
    );


  } catch (error) {

    console.error(
      'Unable to start automation:',
      error
    );


    updateStatus(
      `Automation error: ${error.message}`
    );


    automationButton.disabled =
      false;

    captureButton.disabled =
      false;
  }
}


/*
 * ---------------------------------------------------------
 * Existing GeoJSON → KML
 * ---------------------------------------------------------
 */

function convertGeoJSONToKml(
  geojson
) {

  const kmlParts = [

    '<?xml version="1.0" encoding="UTF-8"?>',

    '<kml xmlns="http://www.opengis.net/kml/2.2">',

    '<Document>',

    '<name>Farmland Boundaries</name>'
  ];


  geojson.features.forEach(
    (
      feature,
      index
    ) => {

      const coords =
        feature.geometry
          .coordinates[0]
          .map(
            coord =>
              `${coord[0]},${coord[1]},0`
          )
          .join(' ');


      const name =
        feature.properties &&
        feature.properties.index != null

          ? `Field ${feature.properties.index}`

          : `Field ${index}`;


      kmlParts.push(

        '<Placemark>',

        `<name>${name}</name>`,

        '<Polygon>',

        '<outerBoundaryIs>',

        '<LinearRing>',

        `<coordinates>${coords}</coordinates>`,

        '</LinearRing>',

        '</outerBoundaryIs>',

        '</Polygon>',

        '</Placemark>'
      );
    }
  );


  kmlParts.push(
    '</Document>',
    '</kml>'
  );


  return kmlParts.join('');
}


/*
 * ---------------------------------------------------------
 * Existing GeoJSON download
 * ---------------------------------------------------------
 */

function downloadGeoJSON() {

  if (!latestGeoJSON) {
    return;
  }


  const blob =
    new Blob(
      [
        JSON.stringify(
          latestGeoJSON,
          null,
          2
        )
      ],
      {
        type:
          'application/json'
      }
    );


  const url =
    URL.createObjectURL(
      blob
    );


  const anchor =
    document.createElement(
      'a'
    );


  anchor.href =
    url;


  anchor.download =
    'farmland_boundaries.geojson';


  anchor.click();


  URL.revokeObjectURL(
    url
  );
}


/*
 * ---------------------------------------------------------
 * Existing KML download
 * ---------------------------------------------------------
 */

function downloadKML() {

  if (!latestGeoJSON) {
    return;
  }


  const kml =
    convertGeoJSONToKml(
      latestGeoJSON
    );


  const blob =
    new Blob(
      [kml],
      {
        type:
          'application/vnd.google-earth.kml+xml'
      }
    );


  const url =
    URL.createObjectURL(
      blob
    );


  const anchor =
    document.createElement(
      'a'
    );


  anchor.href =
    url;


  anchor.download =
    'farmland_boundaries.kml';


  anchor.click();


  URL.revokeObjectURL(
    url
  );
}


/*
 * ---------------------------------------------------------
 * Clear overlay
 * ---------------------------------------------------------
 */

async function clearOverlay() {

  try {

    const [
      tab
    ] =
      await chrome.tabs.query({
        active: true,
        currentWindow: true
      });


    if (!tab) {
      return;
    }


    const result =
      await chrome.scripting.executeScript({

        target: {
          tabId:
            tab.id
        },

        func: () => {

          const existing =
            document.getElementById(
              'farmboundary-overlay-container'
            );


          if (existing) {
            existing.remove();
          }


          return {
            success: true
          };
        }
      });


    if (
      !result ||
      !result[0] ||
      !result[0].result
    ) {

      throw new Error(
        'Unable to clear overlay.'
      );
    }


    updateStatus(
      'Overlay cleared.'
    );


  } catch (error) {

    console.error(
      error
    );


    updateStatus(
      `Error: ${error.message}`
    );
  }
}


/*
 * ---------------------------------------------------------
 * Existing manual Capture & Detect
 *
 * This stage keeps the button available.
 *
 * We will reconnect the full manual workflow
 * after the automation test is confirmed.
 * ---------------------------------------------------------
 */

async function captureTileAndDetect() {

  updateStatus(
    'Manual Capture & Detect is available.'
  );


  console.log(
    'Manual Capture & Detect clicked.'
  );


  /*
   * We intentionally do not modify your
   * existing FastAPI workflow in this
   * debugging step.
   *
   * The next stage will restore/connect
   * the full existing implementation
   * after the background automation is
   * verified.
   */
}


/*
 * ---------------------------------------------------------
 * Button listeners
 * ---------------------------------------------------------
 */

captureButton.addEventListener(
  'click',
  captureTileAndDetect
);


automationButton.addEventListener(
  'click',
  startAutomation
);


clearOverlayButton.addEventListener(
  'click',
  clearOverlay
);


downloadButton.addEventListener(
  'click',
  downloadGeoJSON
);


downloadKmlButton.addEventListener(
  'click',
  downloadKML
);