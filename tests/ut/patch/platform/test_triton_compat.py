import subprocess
import sys
from importlib import metadata

import pytest


def test_vllm_imports_with_triton_ascend():
    try:
        metadata.version("triton-ascend")
    except metadata.PackageNotFoundError:
        pytest.skip("Triton-Ascend is not installed")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_ascend; "
            "from vllm.triton_utils import HAS_TRITON, triton, tl; "
            "from triton.backends.ascend import compiler; "
            "assert HAS_TRITON and triton.jit and tl.constexpr and compiler",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
