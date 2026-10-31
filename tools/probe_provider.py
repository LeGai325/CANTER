"""One authorized model-identity probe. Secrets are read only inside ChatClient."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.provider import ChatClient
from specnet.artifacts import json_text


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/api.glm52.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = ChatClient(json.loads(args.config.read_text()))
    success = False
    try:
        content, record = await client.complete('Return exactly {"answer":"OK","citations":[]}.', 64, "preflight")
        success = json.loads(content).get("answer") == "OK"
    finally:
        result = dict(passed=success, calls=client.records, unconfirmed_remote=client.unconfirmed_remote)
        (args.output / "PREFLIGHT.json").write_text(json_text(result))
        print(json_text(result))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
