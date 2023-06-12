import os
import pytest
import vcr.stubs.httpx_stubs
from tests.support.custom_stub import make_vcr_request
from tytan import symbols, Compile
from tytan.sampler import NQSSampler


@pytest.mark.vcr(
    filter_headers=["x-api-key"],
    match_on=["uri", "method"],
    custom_patches=(
        (vcr.stubs.httpx_stubs, "_make_vcr_request", make_vcr_request),
    ),
)
def test_nqs_sampler_run():
    x, y, z = symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    qubo, offset = Compile(expr).get_qubo()
    api_key = os.environ.get("TYTAN_API_KEY", "foobar")
    sampler = NQSSampler(api_key)
    result = sampler.run(qubo)
    assert result is not None
    assert result["result"] is not None
    assert result["result"]["x"] == 0
    assert result["result"]["y"] == 0
    assert result["result"]["z"] == 0
    assert result["energy"] == 0
    assert result["time"] is not None
