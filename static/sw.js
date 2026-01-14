const CACHE_NAME = 'coachd-v1';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/static/logo.png',
  '/static/logo-icon.png',
  '/static/favicon.svg',
  '/static/favicon.ico',
  '/static/apple-touch-icon.png',
  '/static/web-app-manifest-192x192.png',
  '/static/web-app-manifest-512x512.png',
  '/static/manifest.json'
];

// Install - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Activate - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch - network first for HTML/API, cache first for static assets
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Skip WebSocket and API calls - always go to network
  if (url.pathname.startsWith('/api/') || 
      url.pathname.startsWith('/ws') ||
      url.protocol === 'ws:' ||
      url.protocol === 'wss:') {
    return;
  }
  
  // Static assets - cache first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(request).then((cached) => {
        return cached || fetch(request).then((response) => {
          // Cache new static assets
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return response;
        });
      })
    );
    return;
  }
  
  // HTML pages - network first, fall back to cache
  if (request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Cache successful HTML responses
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }
  
  // Everything else - network first
  event.respondWith(
    fetch(request).catch(() => caches.match(request))
  );
});
