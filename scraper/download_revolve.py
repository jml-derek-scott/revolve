import os
import re
import urllib.parse
from playwright.sync_api import sync_playwright

URL = "https://www.revolve.com/campus-90s-hoodie/dp/FALF-MK113/"
OUTPUT_DIR = "revolve_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with sync_playwright() as p:
    # Disable HTTP/2 to bypass Akamai stream-reset bot detection
    browser = p.chromium.launch(
        headless=True,
        args=["--disable-http2"]
    )
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080}
    )
    page = context.new_page()

    print("Navigating to page...")
    # Use domcontentloaded instead of networkidle
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)

    image_urls = set()

    # Get product ID from URL
    product_code_match = re.search(r'/dp/([A-Z0-9-]+)', URL)
    code = product_code_match.group(1) if product_code_match else "FALF-MK113"

    # Generate potential max-resolution image URLs (V1 to V10)
    for i in range(1, 10):
        image_urls.add(f"https://is4.revolveassets.com/images/p4/n/z/{code}_V{i}.jpg")

    print(f"Downloading high-resolution images for product {code}...")

    downloaded = 0
    for idx, img_url in enumerate(sorted(image_urls), start=1):
        filename = os.path.basename(urllib.parse.urlparse(img_url).path)
        save_path = os.path.join(OUTPUT_DIR, filename)

        try:
            response = page.request.get(img_url)
            if response.status == 200 and len(response.body()) > 5000:  # Ensure it's a real image
                with open(save_path, "wb") as f:
                    f.write(response.body())
                print(f"Saved: {filename}")
                downloaded += 1
        except Exception as e:
            continue

    browser.close()

print(f"Done! Downloaded {downloaded} images to '{OUTPUT_DIR}'.")
