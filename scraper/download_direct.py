import os
import urllib.request

PRODUCT_CODE = "FALF-MK113"
OUTPUT_DIR = "revolve_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Downloading images for {PRODUCT_CODE}...")

# Checks view angles 1 through 10
for i in range(1, 10):
    url = f"https://is4.revolveassets.com/images/p4/n/z/{PRODUCT_CODE}_V{i}.jpg"
    filename = f"{PRODUCT_CODE}_V{i}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                with open(save_path, "wb") as out_file:
                    out_file.write(response.read())
                print(f"Saved: {filename}")
    except urllib.error.HTTPError:
        # 404 indicates no more image variations exist
        continue

print("Done!")
