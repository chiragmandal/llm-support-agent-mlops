# Password Reset (Account Access)

If a user cannot reset their password and sees “token expired”:
1) Ask them to request a new reset email (tokens expire after 15 minutes).
2) Confirm they are using the latest email link (older links invalidate).
3) If they still fail, ask for:
   - their account email
   - approximate reset time
   - screenshot of the error
4) Workaround: use “Try another method” and choose SMS if enabled.
