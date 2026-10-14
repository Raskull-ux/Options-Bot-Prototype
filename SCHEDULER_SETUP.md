# On-time posts: external timer setup (free, about 10 minutes)

## Why

GitHub's free scheduler starts jobs hours late. From the Actions history:

| Job | Scheduled (ET) | Actually started (ET) |
|---|---|---|
| Pre-market post | 9:15 AM | 2:47 PM to 5:42 PM |
| Evening readout | 5:30 PM | 8:26 PM to 10:39 PM |

The bot now protects members from this. A late pre-market job posts nothing, and a late evening readout posts without the @everyone ping. But if you want the posts **on time**, a free outside timer has to start the jobs. cron-job.org does this at no cost.

GitHub's own schedule stays on as a backup. The bot never double-posts: the pre-market post only goes out between 8:00 and 9:35 AM ET, and the evening readout posts once per trading day.

## Step 1: make a GitHub token (5 minutes)

1. GitHub → your profile picture → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. Name: `cron-job-options-bot`. Expiration: 1 year. Put a reminder in your calendar to renew it.
3. Repository access: **Only select repositories** → `Options-Bot-Prototype`.
4. Permissions → Repository permissions → **Actions: Read and write**. Leave everything else as is.
5. Generate, then copy the token. You only see it once. Treat it like a password: never paste it in a chat or a file.

## Step 2: two jobs on cron-job.org (5 minutes)

Sign up free at cron-job.org, then **Create cronjob** twice.

### Job A: Pre-market post

| Field | Value |
|---|---|
| Title | Options bot pre-market |
| URL | `https://api.github.com/repos/Raskull-ux/Options-Bot-Prototype/actions/workflows/premarket.yml/dispatches` |
| Schedule | Custom: Mon to Fri, 09:10 |
| Time zone | America/New_York |

Advanced tab:

| Field | Value |
|---|---|
| Request method | POST |
| Header | `Accept: application/vnd.github+json` |
| Header | `Authorization: Bearer YOUR_TOKEN` |
| Header | `X-GitHub-Api-Version: 2022-11-28` |
| Header | `Content-Type: application/json` |
| Request body | `{"ref":"main"}` |

### Job B: Evening collection + readout

Same as Job A, with these changes:

| Field | Value |
|---|---|
| Title | Options bot evening |
| URL | `https://api.github.com/repos/Raskull-ux/Options-Bot-Prototype/actions/workflows/daily-collect.yml/dispatches` |
| Schedule | Mon to Fri, 16:45 |

4:45 PM leaves room for the free data feed's 15-minute delay after the 4:00 close.

## Step 3: check it worked

- In cron-job.org, use **Test run** on Job A. A good response is **204 No Content**.
- On GitHub → **Actions**, a new **Pre-market post** run should appear within about a minute.
- Outside 8:00 to 9:35 AM, that test run correctly posts nothing. Its log says "outside the window".

## Holidays

Nothing to change. On market holidays the pre-market job sees the market closed and posts nothing. The evening readout sees no new data and posts nothing.
