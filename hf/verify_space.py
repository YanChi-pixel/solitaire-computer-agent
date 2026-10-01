"""Проверяет опубликованный Space живым браузером (Playwright + Chromium).

Открывает демо, ждёт загрузку ONNX-модели, кликает пример карты, читает
предсказание из DOM, собирает ошибки консоли и при желании делает скриншот.
Это сквозная проверка: страница, WASM-инференс, модель, классы, UI.

    pip install playwright && python -m playwright install chromium
    python hf/verify_space.py
    python hf/verify_space.py --namespace WildFuria --shot docs/img/space-demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMESPACE = "WildFuria"
SPACE_NAME = "solitaire-card-recognizer-demo"


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка Space в реальном браузере")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                        help="аккаунт HF, под которым опубликован Space")
    parser.add_argument("--url", default=None, help="проверить произвольный URL")
    parser.add_argument("--shot", default=None, help="куда сохранить скриншот")
    parser.add_argument("--timeout", type=int, default=180,
                        help="сколько секунд ждать загрузку модели")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Нужен playwright: pip install playwright && python -m playwright install chromium")
        return 2

    url = args.url or "https://%s-%s.static.hf.space/" % (
        args.namespace.lower(), SPACE_NAME)
    console_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("console", lambda msg: console_errors.append(msg.text)
                if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append("pageerror: %s" % exc))

        print("открываю:", url)
        page.goto(url, wait_until="load", timeout=90_000)
        page.wait_for_function(
            "() => { const s = document.getElementById('status');"
            " return s && /ready|Error|Could not/.test(s.textContent); }",
            timeout=args.timeout * 1000,
        )
        status = (page.text_content("#status") or "").strip()
        print("статус модели:", status)
        if "ready" not in status.lower():
            print("ПРОВАЛ: модель не загрузилась")
            browser.close()
            return 1

        page.locator(".examples button").first.click()
        page.wait_for_function(
            "() => document.querySelectorAll('#results .bar').length > 0", timeout=90_000)
        page.wait_for_timeout(400)

        print("предсказание (топ-5):")
        for row in page.locator("#results .bar").all():
            print("   %-6s %s" % (row.locator(".name").text_content().strip(),
                                  row.locator(".pct").text_content().strip()))
        print("статус после распознавания:", (page.text_content("#status") or "").strip())

        if console_errors:
            print("ошибки в консоли браузера:")
            for error in console_errors[:5]:
                print("   ", error[:160])
        else:
            print("ошибок в консоли браузера: нет")

        if args.shot:
            shot = Path(args.shot)
            if not shot.is_absolute():
                shot = REPO_ROOT / shot
            shot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shot), full_page=True)
            print("скриншот: %s (%.0f КБ)" % (shot, shot.stat().st_size / 1024))

        browser.close()

    return 1 if console_errors else 0


if __name__ == "__main__":
    sys.exit(main())
