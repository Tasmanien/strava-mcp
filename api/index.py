import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse

app = FastAPI(
    title="Strava Running Coach API",
    description=(
        "Access your Strava activities, stats, and best efforts for AI-powered running coaching. "
        "Use this to analyze training history, plan workouts, estimate race paces, and track progress."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STRAVA_BASE = "https://www.strava.com/api/v3"
STRAVA_AUTH = "https://www.strava.com/oauth/token"
STRAVA_OAUTH = "https://www.strava.com/oauth/authorize"


def _env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"Missing environment variable: {key}")
    return v


async def get_access_token() -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(STRAVA_AUTH, data={
            "client_id": _env("STRAVA_CLIENT_ID"),
            "client_secret": _env("STRAVA_CLIENT_SECRET"),
            "grant_type": "refresh_token",
            "refresh_token": _env("STRAVA_REFRESH_TOKEN"),
        })
        r.raise_for_status()
        return r.json()["access_token"]


async def strava_get(path: str, params: dict | None = None) -> dict | list:
    token = await get_access_token()
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{STRAVA_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=15,
        )
        if r.status_code == 401:
            raise HTTPException(status_code=401, detail="Strava token invalid — re-run /auth to reauthorize")
        r.raise_for_status()
        return r.json()


def verify_key(x_api_key: str = Header(default=None)):
    required = os.environ.get("API_KEY", "")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _pace(seconds_per_km: float) -> str:
    mins = int(seconds_per_km // 60)
    secs = int(seconds_per_km % 60)
    return f"{mins}:{secs:02d}/km"


def _format_activity(a: dict) -> dict:
    dist = a.get("distance", 0)
    moving = a.get("moving_time", 0)
    elapsed = a.get("elapsed_time", 0)
    # Fall back to elapsed when moving_time is implausibly low (bad GPS / paused recording)
    time_for_pace = moving if elapsed == 0 or moving >= elapsed * 0.5 else elapsed
    pace_s = (time_for_pace / (dist / 1000)) if dist > 0 and time_for_pace > 0 else None

    return {
        "id": a["id"],
        "name": a.get("name"),
        "type": a.get("sport_type", a.get("type")),
        "date": a.get("start_date_local"),
        "distance_km": round(dist / 1000, 2),
        "moving_time_min": round(moving / 60, 1),
        "elapsed_time_min": round(elapsed / 60, 1),
        "pace": _pace(pace_s) if pace_s else None,
        "pace_sec_per_km": round(pace_s) if pace_s else None,
        "elevation_gain_m": a.get("total_elevation_gain"),
        "avg_heartrate": a.get("average_heartrate"),
        "max_heartrate": a.get("max_heartrate"),
        "avg_cadence_spm": round(a["average_cadence"] * 2) if a.get("average_cadence") else None,
        "suffer_score": a.get("suffer_score"),
        "perceived_exertion": a.get("perceived_exertion"),
        "calories": a.get("calories"),
        "description": a.get("description") or None,
        "trainer": a.get("trainer", False),
        "commute": a.get("commute", False),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/activities", summary="List activities", tags=["Activities"])
async def list_activities(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(50, ge=1, le=200, description="Results per page, max 200"),
    after: Optional[str] = Query(None, description="ISO date string, e.g. 2024-01-01"),
    before: Optional[str] = Query(None, description="ISO date string, e.g. 2024-12-31"),
    activity_type: Optional[str] = Query(None, description="Filter by type: Run, Ride, Walk, Hike…"),
    _: None = Security(verify_key),
):
    """
    Return a paginated list of activities. For running analysis use activity_type=Run.
    Dates are inclusive. Iterate pages until an empty list is returned to get lifetime data.
    Each activity includes distance, pace, heart rate, elevation, cadence, and suffer score.
    """
    params: dict = {"page": page, "per_page": per_page}
    if after:
        params["after"] = int(datetime.fromisoformat(after).replace(tzinfo=timezone.utc).timestamp())
    if before:
        params["before"] = int(datetime.fromisoformat(before).replace(tzinfo=timezone.utc).timestamp())

    activities = await strava_get("/athlete/activities", params)
    if activity_type:
        activities = [
            a for a in activities
            if a.get("sport_type") == activity_type or a.get("type") == activity_type
        ]
    return [_format_activity(a) for a in activities]


@app.get("/activities/{activity_id}", summary="Get activity detail", tags=["Activities"])
async def get_activity(
    activity_id: int,
    _: None = Security(verify_key),
):
    """
    Full detail for a single activity: splits by km, best efforts (PRs), laps, and gear.
    Best efforts show PR rank (1 = personal best) for distances like 400m, 1km, 1 mile, 5km, 10km, half marathon, marathon.
    """
    a = await strava_get(f"/activities/{activity_id}")
    result = _format_activity(a)

    if a.get("laps"):
        result["laps"] = [
            {
                "lap": l.get("lap_index"),
                "distance_km": round(l.get("distance", 0) / 1000, 2),
                "pace": _pace(l["moving_time"] / (l["distance"] / 1000)) if l.get("distance", 0) > 0 else None,
                "avg_hr": l.get("average_heartrate"),
                "elevation_m": l.get("total_elevation_gain"),
            }
            for l in a["laps"]
        ]

    if a.get("best_efforts"):
        result["best_efforts"] = [
            {
                "distance": e.get("name"),
                "time_sec": e.get("elapsed_time"),
                "pace": _pace(e["elapsed_time"] / (e["distance"] / 1000)) if e.get("distance", 0) > 0 else None,
                "pr_rank": e.get("pr_rank"),
            }
            for e in a["best_efforts"]
        ]

    if a.get("gear") and a["gear"].get("name"):
        result["shoes"] = a["gear"]["name"]

    result["map_has_gps"] = bool(a.get("map", {}).get("summary_polyline"))

    return result


@app.get("/stats", summary="Lifetime and recent stats", tags=["Stats"])
async def get_stats(_: None = Security(verify_key)):
    """
    Athlete profile plus run totals: all-time, year-to-date, and last 4 weeks.
    Use this as the starting point for any coaching conversation to understand current fitness level and history.
    """
    four_weeks_ago = int(datetime.now(timezone.utc).timestamp()) - 28 * 86400

    athlete = await strava_get("/athlete")
    stats, recent_raw = await asyncio.gather(
        strava_get(f"/athletes/{athlete['id']}/stats"),
        strava_get("/athlete/activities", {"after": four_weeks_ago, "per_page": 200}),
    )

    def run_totals(key: str) -> dict:
        t = stats.get(key, {})
        dist = t.get("distance", 0)
        time_s = t.get("moving_time", 0)
        count = t.get("count", 0)
        return {
            "count": count,
            "distance_km": round(dist / 1000, 1),
            "time_hours": round(time_s / 3600, 1),
            "elevation_m": t.get("elevation_gain"),
            "avg_distance_km": round(dist / 1000 / count, 1) if count else None,
        }

    recent_runs = [
        a for a in recent_raw
        if a.get("sport_type") == "Run" or a.get("type") == "Run"
    ]
    recent_dist = sum(a.get("distance", 0) for a in recent_runs)
    recent_time = sum(a.get("moving_time", 0) for a in recent_runs)

    return {
        "athlete": {
            "name": f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip(),
            "city": athlete.get("city"),
            "country": athlete.get("country"),
            "weight_kg": athlete.get("weight"),
            "ftp": athlete.get("ftp"),
        },
        "all_time_runs": run_totals("all_run_totals"),
        "ytd_runs": run_totals("ytd_run_totals"),
        "last_4_weeks_runs": {
            "count": len(recent_runs),
            "distance_km": round(recent_dist / 1000, 1),
            "time_hours": round(recent_time / 3600, 1),
            "avg_distance_km": round(recent_dist / 1000 / len(recent_runs), 1) if recent_runs else None,
        },
        "biggest_climb_m": stats.get("biggest_climb_elevation_gain"),
    }


# ---------------------------------------------------------------------------
# OAuth helpers (run once after deploy to get activity:read_all scope)
# ---------------------------------------------------------------------------

@app.get("/auth", include_in_schema=False)
async def auth_start():
    """Redirect to Strava to authorize with full activity read scope."""
    base = os.environ.get("BASE_URL") or os.environ.get("VERCEL_URL") or "localhost:8000"
    scheme = "http" if "localhost" in base else "https"
    redirect_uri = f"{scheme}://{base}/auth/callback"
    url = (
        f"{STRAVA_OAUTH}"
        f"?client_id={_env('STRAVA_CLIENT_ID')}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=read,activity:read_all"
        f"&approval_prompt=force"
    )
    return RedirectResponse(url)


@app.get("/auth/callback", include_in_schema=False)
async def auth_callback(code: str):
    """Exchange OAuth code for tokens. Copy the refresh_token to your Vercel env vars."""
    async with httpx.AsyncClient() as client:
        r = await client.post(STRAVA_AUTH, data={
            "client_id": _env("STRAVA_CLIENT_ID"),
            "client_secret": _env("STRAVA_CLIENT_SECRET"),
            "code": code,
            "grant_type": "authorization_code",
        })
        r.raise_for_status()
        data = r.json()

    athlete = data.get("athlete", {})
    name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
    new_token = data.get("refresh_token", "")
    scope = data.get("scope", "")

    html = f"""
    <html><body style="font-family:monospace;padding:2em;max-width:700px">
    <h2>✅ Strava authorization successful</h2>
    <p>Athlete: <strong>{name}</strong></p>
    <p>Scope: <code>{scope}</code></p>
    <p>Copy this refresh token to your <strong>Vercel environment variables</strong> as <code>STRAVA_REFRESH_TOKEN</code>:</p>
    <pre style="background:#f0f0f0;padding:1em;word-break:break-all">{new_token}</pre>
    <p>Then redeploy in Vercel dashboard (or it picks up env changes automatically on next request).</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
