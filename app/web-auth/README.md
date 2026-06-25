# CarSee — browser auth pages (Supabase)

Static pages for Supabase auth emails on mobile.

| Page | URL |
|------|-----|
| Welcome (after signup) | `https://carsee-auth.onrender.com/welcome.html` |
| Password reset | `https://carsee-auth.onrender.com/reset-password.html` |
| Email confirmation | `https://carsee-auth.onrender.com/confirm-email.html` |

## Render Static Site

| Setting | Value |
|---------|--------|
| Repository | `Icestreamm/carsee` |
| Root Directory | `app/web-auth` |
| Publish Directory | `.` |
| Build Command | *(empty)* |

Add all URLs (+ trailing-slash variants) in Supabase → Authentication → URL Configuration → Redirect URLs.

Set **Site URL** to `https://carsee-auth.onrender.com/welcome.html` (not the raw `*.supabase.co` URL — that shows a JSON error in the browser).
