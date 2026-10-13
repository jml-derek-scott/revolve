import argparse
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_product_code(input_str: str) -> str:
    # Strip query parameters and trailing whitespace
    clean_str = input_str.split("?")[0].strip()
    
    # Extract code from /dp/CODE/ pattern
    match = re.search(r'/dp/([A-Za-z0-9-]+)', clean_str)
    if match:
        return match.group(1).upper()
    
    # Fallback if raw product code was passed (e.g. LIOR-WO130)
    if re.match(r'^[A-Za-z0-9]+-[A-Za-z0-9]+$', clean_str):
        return clean_str.upper()
        
    raise ValueError(f"Could not extract product code from: '{input_str}'")

def download_image(product_code: str, index: int, output_dir: str) -> bool:
    url = f"https://is4.revolveassets.com/images/p4/n/z/{product_code}_V{index}.jpg"
    filename = f"{product_code}_V{index}.jpg"
    save_path = os.path.join(output_dir, filename)

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = response.read()
                if len(data) > 5000:  # Skip placeholder/error images
                    with open(save_path, "wb") as f:
                        f.write(data)
                    print(f"  ✓ Saved: {filename}")
                    return True
    except urllib.error.HTTPError:
        pass  # 404 indicates no image at this index
    except Exception as e:
        print(f"  ✗ Error downloading {filename}: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Download high-resolution images from Revolve.")
    parser.add_argument(
        "target", 
        help="Product path, URL, or code (e.g., /dp/LIOR-WO130/ or full URL)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="revolve_images", 
        help="Output directory (default: revolve_images)"
    )
    
    args = parser.parse_args()

    try:
        product_code = extract_product_code(args.target)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Product Code: {product_code}")
    print(f"Output Directory: {args.output}")
    print("Downloading images...")

    os.makedirs(args.output, exist_ok=True)

    # Check views V1 through V15 concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_image, product_code, i, args.output)
            for i in range(1, 16)
        ]
        results = [f.result() for f in futures]

    print(f"Done! Downloaded {sum(results)} images.")

if __name__ == "__main__":
    main()
