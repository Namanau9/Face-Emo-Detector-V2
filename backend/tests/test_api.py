import argparse
import pathlib

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the emotion API.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL.")
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    print("Health:", health.json())

    with image_path.open("rb") as handle:
        response = requests.post(
            f"{args.base_url}/predict",
            files={"file": (image_path.name, handle, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    print("Prediction:", response.json())


if __name__ == "__main__":
    main()
