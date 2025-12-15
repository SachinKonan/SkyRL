import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Initialize Ray once for the entire test session."""
    if not ray.is_initialized():
        # Exclude large directories from runtime env to speed up tests
        # data/ contains large ChromaDB files (60GB+)
        ray.init(
            runtime_env={
                "excludes": [
                    "exports/",
                    "outputs/",
                    "logs/",
                    "wandb/",
                    "data/",
                    "core.*",
                ]
            }
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
