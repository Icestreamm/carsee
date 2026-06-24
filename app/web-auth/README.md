# CarSee — browser auth pages (Supabase)

Static pages for Supabase auth emails on mobile.

| Page | URL |
|------|-----|
| Password reset | `https://carsee-auth.onrender.com/reset-password.html` |
| Email confirmation | `https://carsee-auth.onrender.com/confirm-email.html` |

## Render Static Site

| Setting | Value |
|---------|--------|
| Repository | `Icestreamm/carsee` |
| Root Directory | `app/web-auth` |
| Publish Directory | `.` |
| Build Command | *(empty)* |

Add both URLs (+ trailing-slash variants) in Supabase → Authentication → URL Configuration → Redirect URLs.
