import subprocess
import sys
import textwrap


def test_importing_trainer_does_not_initialize_deepspeed():
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys
        import types

        fake = types.ModuleType("deepspeed")
        fake.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed", loader=None
        )
        fake.__version__ = "0.0.test"

        def init_distributed(*args, **kwargs):
            raise RuntimeError("deepspeed init called during import")

        fake.init_distributed = init_distributed
        sys.modules["deepspeed"] = fake

        import qwen3_omni_pretrain.training.trainer_thinker
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
