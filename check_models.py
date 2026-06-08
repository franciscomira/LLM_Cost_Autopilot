import httpx
import os
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("GITHUB_TOKEN", "")
if not token or token == "github_pat_...":
    print("ERROR: GITHUB_TOKEN not set in .env")
    exit(1)

r = httpx.get(
    "https://models.inference.ai.azure.com/models",
    headers={"Authorization": f"Bearer {token}"},
)
print(f"Status: {r.status_code}\n")

models = r.json()
if isinstance(models, list):
    for m in sorted(models, key=lambda x: x.get("name", "")):
        print(f"{m.get('name',''):<45} {m.get('model_family','')}")
else:
    print(models)
