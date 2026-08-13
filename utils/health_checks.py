"""Runtime health checks for distributed training."""

import os


def distributed_env_health() -> dict:
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    missing = [k for k in required if k not in os.environ]
    return {"ok": len(missing) == 0, "missing": missing}
