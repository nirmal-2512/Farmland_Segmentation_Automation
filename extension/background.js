console.log('Farmland Boundary Detector background service worker loaded.');

const FASTAPI_URL = 'http://localhost:8000';

const TEST_TILE = {
  tile_id: 'tile_0000_0000',

  center: {
    lat: 24.281198846744214,
    lon: 75.57792560550555
  },

  bounds: {
    north: 24.28346607346837,
    south: 24.27893159536946,
    east: 75.58039906550955,
    west: 75.57545223305652
  }
};

const TARGET_ZOOM = 18;

const CENTER_TOLERANCE = 0.00005;

const PAGE_LOAD_WAIT_MS = 5000;

const IMAGERY_WAIT_MS = 5000;

const MAX_NAVIGATION_ATTEMPTS = 3;


/*
 * ============================================================
 * INSTALLATION
 * ============================================================
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log(
    'Farmland Boundary Detector installed.'
  );
});


/*
 * ============================================================
 * STATUS
 * ============================================================
 */

function sendAutomationStatus(
  status,
  message,
  extra = {}
) {
  console.log(
    `[Automation] ${status}: ${message}`
  );

  chrome.runtime.sendMessage({
    type: 'automation-status',
    status,
    message,
    ...extra
  }).catch(() => {
    /*
     * Popup may already be closed.
     * This is expected.
     */
  });
}


/*
 * ============================================================
 * SLEEP
 * ============================================================
 */

function sleep(ms) {
  return new Promise(
    resolve => setTimeout(resolve, ms)
  );
}


/*
 * ============================================================
 * WAIT FOR TAB LOAD
 * ============================================================
 */

function waitForTabLoad(tabId) {
  return new Promise(resolve => {

    let finished = false;

    let timeout = null;


    function finish() {

      if (finished) {
        return;
      }

      finished = true;

      if (timeout) {
        clearTimeout(timeout);
      }

      chrome.tabs.onUpdated.removeListener(
        listener
      );

      resolve();
    }


    function listener(
      updatedTabId,
      changeInfo
    ) {

      if (
        updatedTabId === tabId &&
        changeInfo.status === 'complete'
      ) {
        finish();
      }
    }


    chrome.tabs.onUpdated.addListener(
      listener
    );


    timeout = setTimeout(() => {

      console.warn(
        'Google Maps page-load timeout. Continuing.'
      );

      finish();

    }, 20000);
  });
}


/*
 * ============================================================
 * EXECUTE SCRIPT IN MAP TAB
 * ============================================================
 */

async function executeTabScript(
  tabId,
  func,
  args = []
) {

  try {

    const results =
      await chrome.scripting.executeScript({
        target: {
          tabId
        },
        func,
        args
      });


    if (
      !results ||
      !results[0]
    ) {

      return {
        success: false,
        message: 'No script result.'
      };
    }


    return results[0].result;

  } catch (error) {

    return {
      success: false,
      message:
        error.message ||
        String(error)
    };
  }
}


/*
 * ============================================================
 * GET MAP STATE
 *
 * We intentionally read the URL and viewport after navigation.
 * This allows us to verify that Google Maps actually moved to
 * the requested tile.
 * ============================================================
 */

async function getMapState(tabId) {

  return executeTabScript(
    tabId,

    () => {

      function parseUrlCenterZoom(url) {

        /*
         * Standard @lat,lng,zoomz format
         */

        const atMatch =
          url.match(
            /@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(?:,(\d+(?:\.\d+)?)z)?/
          );


        if (atMatch) {

          return {
            lat:
              parseFloat(
                atMatch[1]
              ),

            lng:
              parseFloat(
                atMatch[2]
              ),

            zoom:
              parseFloat(
                atMatch[3]
              )
          };
        }


        /*
         * API=1 map URL.
         *
         * Example:
         *
         * /maps/@?api=1&map_action=map
         * &center=24.28,75.57
         * &zoom=18
         */

        const centerMatch =
          url.match(
            /[?&]center=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/
          );


        const zoomMatch =
          url.match(
            /[?&]zoom=(\d+(?:\.\d+)?)/
          );


        if (
          centerMatch &&
          zoomMatch
        ) {

          return {

            lat:
              parseFloat(
                centerMatch[1]
              ),

            lng:
              parseFloat(
                centerMatch[2]
              ),

            zoom:
              parseFloat(
                zoomMatch[1]
              )
          };
        }


        /*
         * Google Maps query parameters.
         */

        const llMatch =
          url.match(
            /[?&](?:center|ll)=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/
          );


        const queryZoomMatch =
          url.match(
            /[?&](?:zoom|z)=(\d+(?:\.\d+)?)/
          );


        if (
          llMatch &&
          queryZoomMatch
        ) {

          return {

            lat:
              parseFloat(
                llMatch[1]
              ),

            lng:
              parseFloat(
                llMatch[2]
              ),

            zoom:
              parseFloat(
                queryZoomMatch[1]
              )
          };
        }


        /*
         * Google Maps place URL fallback.
         */

        const gmapMatch =
          url.match(
            /!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/
          );


        if (gmapMatch) {

          return {

            lat:
              parseFloat(
                gmapMatch[1]
              ),

            lng:
              parseFloat(
                gmapMatch[2]
              ),

            zoom: 18
          };
        }


        /*
         * Hash format.
         */

        const hashMatch =
          url.match(
            /#@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z/
          );


        if (hashMatch) {

          return {

            lat:
              parseFloat(
                hashMatch[1]
              ),

            lng:
              parseFloat(
                hashMatch[2]
              ),

              zoom:
                atMatch[3]
                  ? parseFloat(atMatch[3])
                  : null
          };
        }


        return null;
      }


      function getMapRect() {

        const canvasCandidates =
          Array.from(
            document.querySelectorAll(
              'canvas'
            )
          )
          .map(canvas => ({
            canvas,
            rect:
              canvas.getBoundingClientRect()
          }))
          .filter(
            entry =>
              entry.rect.width > 200 &&
              entry.rect.height > 200
          );


        if (
          canvasCandidates.length
        ) {

          const best =
            canvasCandidates.reduce(
              (previous, current) => {

                const previousArea =
                  previous.rect.width *
                  previous.rect.height;


                const currentArea =
                  current.rect.width *
                  current.rect.height;


                return currentArea >
                  previousArea
                  ? current
                  : previous;
              }
            );


          const rect =
            best.rect;


          const searchBox =
            document.querySelector(
              'input[aria-label="Search Google Maps"], #searchboxinput'
            );


          let sidebarWidth = 80;


          if (searchBox) {

            const panel =
              searchBox.closest(
                '[role="main"]'
              ) ||
              searchBox.closest(
                '[jsaction]'
              ) ||
              searchBox.parentElement
                ?.parentElement
                ?.parentElement;


            if (panel) {

              const panelRect =
                panel.getBoundingClientRect();


              if (
                panelRect.right > 0 &&
                panelRect.right <
                  window.innerWidth * 0.4
              ) {

                sidebarWidth =
                  Math.ceil(
                    panelRect.right
                  );
              }
            }
          }


          return {

            left:
              sidebarWidth,

            top:
              rect.top,

            width:
              Math.max(
                1,
                rect.width -
                sidebarWidth
              ),

            height:
              Math.max(
                1,
                rect.height
              )
          };
        }


        return {

          left: 80,

          top: 0,

          width:
            Math.max(
              1,
              window.innerWidth - 80
            ),

          height:
            Math.max(
              1,
              window.innerHeight
            )
        };
      }


      const url =
        window.location.href;


      const centerZoom =
        parseUrlCenterZoom(url);


      const mapRect =
        getMapRect();


      if (!centerZoom) {

        return {

          success: false,

          message:
            'Could not determine map center from URL.',

          url
        };
      }


      return {

        success: true,

        url,

        center: {

          lat:
            centerZoom.lat,

          lng:
            centerZoom.lng
        },

        zoom:
          centerZoom.zoom,

        mapRect,

        physicalWidth:
          Math.round(
            mapRect.width *
            (window.devicePixelRatio || 1)
          ),

        physicalHeight:
          Math.round(
            mapRect.height *
            (window.devicePixelRatio || 1)
          ),

        devicePixelRatio:
          window.devicePixelRatio || 1,

        pageTitle:
          document.title
      };
    }
  );
}


/*
 * ============================================================
 * DISTANCE / CENTER VALIDATION
 * ============================================================
 */

function getCenterDifference(
  actual,
  target
) {

  return {

    latitude:
      Math.abs(
        actual.lat -
        target.lat
      ),

    longitude:
      Math.abs(
        actual.lng -
        target.lon
      )
  };
}


function isCenterCorrect(
  actual,
  target
) {

  if (
    !actual ||
    typeof actual.lat !== 'number' ||
    typeof actual.lng !== 'number'
  ) {
    return false;
  }


  const difference =
    getCenterDifference(
      actual,
      target
    );


  return (
    difference.latitude <=
      CENTER_TOLERANCE &&
    difference.longitude <=
      CENTER_TOLERANCE
  );
}


/*
 * ============================================================
 * BUILD EXACT GOOGLE MAPS URL
 * ============================================================
 *
 * Google Maps supports:
 *
 * center
 * zoom
 * basemap=satellite
 *
 * This is more explicit than the old @lat,lng,18z URL.
 * ============================================================
 */

function buildMapUrl(tile) {

  const center =
    `${tile.center.lat},${tile.center.lon}`;


  return (
    'https://www.google.com/maps/@?' +
    'api=1' +
    '&map_action=map' +
    `&center=${encodeURIComponent(center)}` +
    `&zoom=${TARGET_ZOOM}` +
    '&basemap=satellite'
  );
}


/*
 * ============================================================
 * NAVIGATE TO TILE AND VERIFY
 * ============================================================
 */

async function navigateToTile(
  tabId,
  tile
) {

  const url =
    buildMapUrl(
      tile
    );


  console.log(
    'Navigation URL:',
    url
  );


  for (
    let attempt = 1;
    attempt <= MAX_NAVIGATION_ATTEMPTS;
    attempt++
  ) {

    sendAutomationStatus(
      'navigating',
      `Navigating to ${tile.tile_id} (attempt ${attempt}/${MAX_NAVIGATION_ATTEMPTS})...`
    );


    /*
     * IMPORTANT:
     *
     * Make the target tab active.
     *
     * captureVisibleTab captures the visible
     * tab in the window, so we must ensure that
     * the target map tab is the active tab.
     */

    await chrome.tabs.update(
      tabId,
      {
        active: true,
        url
      }
    );


    sendAutomationStatus(
      'loading',
      'Waiting for Google Maps page to load...'
    );


    await waitForTabLoad(
      tabId
    );


    sendAutomationStatus(
      'waiting',
      'Waiting for map rendering and satellite imagery...'
    );


    await sleep(
      PAGE_LOAD_WAIT_MS +
      IMAGERY_WAIT_MS
    );


    const mapState =
      await getMapState(
        tabId
      );


    console.log(
      `Map state after attempt ${attempt}:`,
      mapState
    );


    if (
      mapState &&
      mapState.success
    ) {

      const difference =
        getCenterDifference(
          mapState.center,
          tile.center
        );


      console.log(
        'Requested center:',
        tile.center
      );


      console.log(
        'Actual center:',
        mapState.center
      );


      console.log(
        'Center difference:',
        difference
      );


      if (
        isCenterCorrect(
          mapState.center,
          tile.center
        )
      ) {

        sendAutomationStatus(
          'verified',
          `Map verified at ${tile.tile_id}`
        );


        return {
          success: true,
          mapState,
          attempts: attempt
        };
      }


      sendAutomationStatus(
        'retrying',
        `Map is not at requested tile. Difference: lat=${difference.latitude}, lon=${difference.longitude}`
      );

    } else {

      sendAutomationStatus(
        'retrying',
        'Could not read map center. Retrying navigation.'
      );
    }


    await sleep(
      2000
    );
  }


  return {

    success: false,

    message:
      'Unable to verify that Google Maps reached the requested tile.'
  };
}


/*
 * ============================================================
 * CAPTURE SCREENSHOT
 * ============================================================
 */

async function captureScreenshot(
  windowId
) {

  sendAutomationStatus(
    'capturing',
    'Capturing screenshot...'
  );


  const screenshot =
    await chrome.tabs.captureVisibleTab(
      windowId,
      {
        format: 'png'
      }
    );


  if (!screenshot) {

    throw new Error(
      'Screenshot capture returned empty result.'
    );
  }


  return screenshot;
}


/*
 * ============================================================
 * UPLOAD SCREENSHOT TO FASTAPI
 * ============================================================
 */

async function uploadScreenshot(
  screenshotDataUrl,
  tile,
  mapState
) {

  sendAutomationStatus(
    'saving',
    'Uploading screenshot to FastAPI...'
  );


  const response =
    await fetch(
      screenshotDataUrl
    );


  const blob =
    await response.blob();


  const formData =
    new FormData();


  formData.append(
    'file',
    blob,
    `${tile.tile_id}.png`
  );


  formData.append(
    'tile_id',
    tile.tile_id
  );


  formData.append(
    'job_id',
    'garoth_test'
  );


  formData.append(
    'center_lat',
    String(
      tile.center.lat
    )
  );


  formData.append(
    'center_lon',
    String(
      tile.center.lon
    )
  );


  formData.append(
    'map_lat',
    String(
      mapState.center.lat
    )
  );


  formData.append(
    'map_lon',
    String(
      mapState.center.lng
    )
  );


  formData.append(
    'zoom',
    String(
      mapState.zoom
    )
  );


  formData.append(
    'north',
    String(
      tile.bounds.north
    )
  );


  formData.append(
    'south',
    String(
      tile.bounds.south
    )
  );


  formData.append(
    'east',
    String(
      tile.bounds.east
    )
  );


  formData.append(
    'west',
    String(
      tile.bounds.west
    )
  );


  formData.append(
    'map_width',
    String(
      mapState.physicalWidth
    )
  );


  formData.append(
    'map_height',
    String(
      mapState.physicalHeight
    )
  );


  const saveResponse =
    await fetch(
      `${FASTAPI_URL}/automation/save-screenshot`,
      {
        method: 'POST',
        body: formData
      }
    );


  if (!saveResponse.ok) {

    const errorText =
      await saveResponse.text();


    throw new Error(
      `Screenshot save failed: ${saveResponse.status} ${errorText}`
    );
  }


  const result =
    await saveResponse.json();


  if (
    !result.success
  ) {

    throw new Error(
      result.message ||
      'FastAPI did not confirm screenshot save.'
    );
  }


  return result;
}


/*
 * ============================================================
 * SINGLE TILE AUTOMATION
 * ============================================================
 */

async function runSingleTileAutomation(
  tabId,
  windowId
) {

  try {

    sendAutomationStatus(
      'starting',
      `Starting ${TEST_TILE.tile_id}`
    );


    /*
     * STEP 1
     * Navigate and VERIFY.
     */

    const navigationResult =
      await navigateToTile(
        tabId,
        TEST_TILE
      );


    if (
      !navigationResult.success
    ) {

      throw new Error(
        navigationResult.message
      );
    }


    const mapState =
      navigationResult.mapState;


    /*
     * STEP 2
     * Capture only after the map
     * has been verified.
     */

    const screenshot =
      await captureScreenshot(
        windowId
      );


    /*
     * STEP 3
     * Upload automatically.
     */

    const saveResult =
      await uploadScreenshot(
        screenshot,
        TEST_TILE,
        mapState
      );


    /*
     * STEP 4
     * Store test metadata in extension storage.
     */

    const automationResult = {

      success: true,

      tile:
        TEST_TILE,

      mapState,

      screenshot:
        saveResult,

      timestamp:
        new Date().toISOString()
    };


    await chrome.storage.local.set({

      automationTest:
        automationResult
    });


    sendAutomationStatus(
      'completed',
      `Screenshot saved successfully for ${TEST_TILE.tile_id}`,
      {
        result:
          automationResult
      }
    );


    console.log(
      '========================================'
    );

    console.log(
      'SINGLE TILE AUTOMATION COMPLETE'
    );

    console.log(
      '========================================'
    );

    console.log(
      automationResult
    );


    return automationResult;


  } catch (error) {

    console.error(
      'Single tile automation failed:',
      error
    );


    sendAutomationStatus(
      'error',
      error.message ||
      String(error)
    );


    return {

      success: false,

      error:
        error.message ||
        String(error)
    };
  }
}


/*
 * ============================================================
 * MESSAGE HANDLER
 * ============================================================
 */

chrome.runtime.onMessage.addListener(
  (
    message,
    sender,
    sendResponse
  ) => {

    if (
      !message ||
      !message.type
    ) {
      return;
    }


    if (
      message.type ===
      'start-single-tile-automation'
    ) {

      chrome.tabs.query({
        active: true,
        currentWindow: true
      })
      .then(
        async tabs => {

          const tab =
            tabs[0];


          if (!tab) {

            sendAutomationStatus(
              'error',
              'No active tab found.'
            );

            return;
          }


          if (
            !tab.url ||
            (
              !tab.url.includes(
                'google.com/maps'
              ) &&
              !tab.url.includes(
                'earth.google.com'
              ) &&
              !tab.url.includes(
                'google.com/earth'
              )
            )
          ) {

            sendAutomationStatus(
              'error',
              'Please open Google Maps first.'
            );

            return;
          }


          await runSingleTileAutomation(
            tab.id,
            tab.windowId
          );
        }
      );


      sendResponse({
        success: true,
        message:
          'Automation started in background.'
      });


      return true;
    }
  }
);


/*
 * ============================================================
 * EXISTING ACTION HANDLER
 * ============================================================
 */

chrome.action.onClicked.addListener(
  tab => {

    if (
      !tab.url ||
      !tab.url.includes(
        'google.com/maps'
      )
    ) {

      chrome.notifications.create({
        type: 'basic',

        iconUrl:
          'icon48.png',

        title:
          'Farmland Boundary Detector',

        message:
          'Please open Google Maps first.'
      });
    }
  }
);