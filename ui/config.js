/* Retake UI runtime config.
   RETAKE_API_BASE: origin of the Retake API, no trailing slash.
   "" = same origin (the FastAPI app serves ui/ at "/").
   deploy/pages-publish.sh rewrites this line for GitHub Pages, e.g.
   window.RETAKE_API_BASE = "https://retake-api.example.com";
   A value saved in the Settings drawer (localStorage) overrides this. */
window.RETAKE_API_BASE = window.RETAKE_API_BASE || "";
