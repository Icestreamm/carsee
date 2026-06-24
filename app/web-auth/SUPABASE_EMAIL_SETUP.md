# Supabase email templates — browser password reset (required)

## Why this is required

Default Supabase reset emails use PKCE (`token=pkce_...` via `supabase.co/auth/v1/verify`).
The PKCE **code verifier lives in the mobile app**, but the user opens the link in **Chrome/Safari**.
The browser cannot finish PKCE → you see **"invalid or expired"**.

**Fix:** point the reset email directly to our browser page with `token_hash` (works in any browser, no app needed).

---

## Step 1 — Supabase email template (5 min)

1. Open [Supabase Dashboard](https://supabase.com) → project **tbwfotarskyrzkmisxcf**
2. **Authentication** → **Email Templates** → **Reset password**
3. Replace the body with the contents of:
   `app/web-auth/supabase-reset-password-email.html`
4. **Save**

The button link must be exactly:

```
https://carsee-auth.onrender.com/reset-password.html?token_hash={{ .TokenHash }}&type=recovery
```

Do **not** use `{{ .ConfirmationURL }}` for mobile reset.

---

## Step 2 — Redirect URLs (already should be set)

**Authentication** → **URL Configuration** → **Redirect URLs**:

```
https://carsee-auth.onrender.com/reset-password.html
https://carsee-auth.onrender.com/reset-password.html/
```

---

## Step 3 — Render deploy

Redeploy `carsee-auth` static site (latest `app/web-auth` from GitHub).

---

## User flow after setup

1. User taps **Change Password** in CarSee app → email sent
2. User opens email on phone → taps **Reset Password**
3. Browser opens `carsee-auth.onrender.com` → password form
4. User sets new password **in the browser**
5. User opens CarSee app → signs in with new password

No app deep link required.

---

## Email rate limit

Built-in Supabase email = **2 emails/hour**. For production, set up **Custom SMTP** (Resend, SendGrid, etc.) under **Project Settings → Authentication → SMTP**.
