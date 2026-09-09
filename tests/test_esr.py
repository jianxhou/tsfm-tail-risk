import pytest

pytestmark = pytest.mark.skip(
    reason="ESR backtest pending: esreg port not yet implemented; reference rows "
           "frozen in r_references.csv (esback_esr1..3). Testing policy: no "
           "scientific use until the reference checks are green. See backtests/esr.py."
)


def test_esr_placeholder():
    raise AssertionError("unreachable while module is pending")
