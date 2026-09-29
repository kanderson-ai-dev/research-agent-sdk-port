"""Record the README demo GIF against a fully deterministic app.

Boots the same deterministic in-process app as ``capture_screenshots.py``
(eval corpus + ``StubModel`` + HITL review enabled) and drives the real
UI slowly enough to be watchable: landing → console → type the topic →
live progress → HITL modal → approve → finished report. Frames are
captured as screenshots (~8fps) and assembled into
``docs/screenshots/demo.gif`` with Pillow — zero network, zero secrets.

    uv run --with playwright python scripts/record_demo.py
"""

import asyncio
import io
import tempfile
from dataclasses import replace
from pathlib import Path

import uvicorn
from PIL import Image
from playwright.async_api import async_playwright

from app.core.config import Settings
from app.main import create_app
from evaluation.harness import build_eval_deps, load_dataset

BASE = "http://127.0.0.1:8124"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
FRAME_MS = 125  # ~8fps
GIF_WIDTH = 1100


async def _record_frames(page, frames: list[bytes], stop: asyncio.Event) -> None:
    """Capture page screenshots into ``frames`` until ``stop`` is set."""
    while not stop.is_set():
        frames.append(await page.screenshot(type="png"))
        await asyncio.sleep(FRAME_MS / 1000)


async def main() -> None:
    """Record the console flow and assemble the GIF."""
    spec = load_dataset()["topics"][0]
    deps = replace(build_eval_deps(spec), require_human_review=True)
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as db:
        db_path = db.name
    settings = Settings(_env_file=None, database_url=f"sqlite:///{db_path}")
    app = create_app(settings=settings, deps=deps)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8124, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    frames: list[bytes] = []
    stop = asyncio.Event()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1280, "height": 820})
            recorder = asyncio.create_task(_record_frames(page, frames, stop))

            await page.goto(f"{BASE}/")
            await page.wait_for_timeout(1500)
            await page.click("a[href='/console/']")
            await page.wait_for_selector("#research-panel:not(.hidden)")
            await page.wait_for_timeout(800)

            await page.locator("#topic-input").press_sequentially(
                spec["topic"], delay=55
            )
            await page.wait_for_timeout(600)
            await page.click("#topic-form button[type=submit]")

            await page.wait_for_selector("#progress-panel:not(.hidden)")
            await page.wait_for_timeout(2500)
            await page.wait_for_selector(
                "#review-panel:not(.hidden)", timeout=30_000
            )
            await page.wait_for_timeout(1800)

            await page.click("#approve-btn")
            await page.wait_for_selector(
                "#result-panel:not(.hidden)", timeout=30_000
            )
            await page.wait_for_timeout(1200)
            for _ in range(5):
                await page.mouse.wheel(0, 380)
                await page.wait_for_timeout(350)
            await page.wait_for_timeout(900)

            stop.set()
            await recorder
            await browser.close()
    finally:
        server.should_exit = True
        await task

    images = [Image.open(io.BytesIO(f)) for f in frames]
    ratio = GIF_WIDTH / images[0].width
    size = (GIF_WIDTH, int(images[0].height * ratio))
    frames_img = [im.convert("RGB").resize(size) for im in images]
    gif = OUT / "demo.gif"
    frames_img[0].save(
        gif,
        save_all=True,
        append_images=frames_img[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"GIF written to {gif} — {len(frames_img)} frames, {gif.stat().st_size // 1024} KB")


if __name__ == "__main__":
    asyncio.run(main())
