"""Capture real README screenshots with a fully deterministic app.

Boots the FastAPI app in-process (uvicorn server task) wired to the eval
corpus + ``StubModel`` + HITL review enabled, drives the real console with
headless Chromium, and saves PNGs under ``docs/screenshots/``. Zero
network, zero secrets — the job runs on fixture data only.

    uv run --with playwright python scripts/capture_screenshots.py
"""

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path

import uvicorn
from playwright.async_api import async_playwright

from app.core.config import Settings
from app.main import create_app
from evaluation.harness import build_eval_deps, load_dataset

BASE = "http://127.0.0.1:8123"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


async def main() -> None:
    """Run the deterministic app and capture the three README shots."""
    spec = load_dataset()["topics"][0]
    deps = replace(build_eval_deps(spec), require_human_review=True)
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as db:
        db_path = db.name
    settings = Settings(_env_file=None, database_url=f"sqlite:///{db_path}")
    app = create_app(settings=settings, deps=deps)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            await page.goto(f"{BASE}/")
            await page.wait_for_timeout(400)
            await page.screenshot(path=str(OUT / "landing.png"))

            await page.goto(f"{BASE}/console/")
            await page.wait_for_selector("#research-panel:not(.hidden)")
            await page.fill("#topic-input", spec["topic"])
            await page.click("#topic-form button[type=submit]")
            await page.wait_for_selector(
                "#review-panel:not(.hidden)", timeout=30_000
            )
            await page.wait_for_timeout(300)
            await page.screenshot(path=str(OUT / "console-review.png"))

            await page.click("#approve-btn")
            await page.wait_for_selector(
                "#result-panel:not(.hidden)", timeout=30_000
            )
            await page.wait_for_timeout(400)
            await page.screenshot(path=str(OUT / "console-report.png"))
            await browser.close()
    finally:
        server.should_exit = True
        await task
    print(f"Screenshots written to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
