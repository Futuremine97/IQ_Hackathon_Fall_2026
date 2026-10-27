"""End-to-end check of the Q-BREW MCP server: starts it over stdio, lists the tools and calls each
one on frozen data (no internet). Also checks that the server's QUBO equals the study's
build_cost and that a stronger coffee never gives a lower-variance exact optimum.

Run from the repository root:  python mcp_server/smoke_test.py
"""
import asyncio
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from mcp import Client, StdioServerParameters  # noqa: E402

import coffee_portfolio_server as srv  # noqa: E402
from qps.quantum import bit_matrix  # noqa: E402
from run_qaoa_portfolio import build_cost  # noqa: E402

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]


def payload(res):
    assert not res.is_error, res.content[0].text
    if res.structured_content is not None:
        d = res.structured_content
        return d.get("result", d) if set(d) == {"result"} else d
    return json.loads(res.content[0].text)


async def main():
    # 1. the server's QUBO is the study's QUBO at gamma = 1, k = 4, 18 assets
    rng = np.random.default_rng(1)
    mu = rng.standard_normal(18)
    a = rng.standard_normal((40, 18))
    S = a.T @ a / 40
    B = bit_matrix(18).astype(float)
    diff = np.max(np.abs(srv.qubo_energies(mu, S, B, 4, 1.0) - build_cost(mu, S, B)[0]))
    assert diff < 1e-12, diff
    print(f"QUBO == run_qaoa_portfolio.build_cost (max |diff| {diff:.1e})")

    params = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "coffee_portfolio_server.py")])
    async with Client(params, read_timeout_seconds=600) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        assert names == ["brew_portfolio", "list_universe", "prophet_forecast", "qsvm_rank", "study_results"], names
        print("tools:", names)

        u = payload(await client.call_tool("list_universe", {}))
        assert len(u["universe"]) == 18 and u["last_date"] == "2026-10-16", u
        s = payload(await client.call_tool("study_results", {}))
        print("study:", s["portfolio"]["verdict"], "Sharpe QAOA", round(s["portfolio"]["sharpe_qaoa"], 2),
              "vs equal weight", round(s["portfolio"]["sharpe_equal_weight"], 2))

        t0 = time.time()
        f = payload(await client.call_tool("prophet_forecast", {"ticker": "SPY", "as_of": "2026-09-30"}))
        assert f["interval_80"][0] <= f["forecast_return"] <= f["interval_80"][1], f
        print(f"prophet_forecast SPY {f['as_of']}: {100 * f['forecast_return']:+.2f} % {f['signal']} ({time.time() - t0:.1f} s)")

        t0 = time.time()
        q = payload(await client.call_tool("qsvm_rank", {"as_of": "2026-08-31"}))
        assert sorted(r["qsvm_rank"] for r in q["ranking"]) == list(range(1, 19)), q
        print(f"qsvm_rank {q['as_of']}: top-3 {q['predicted_top_half_qsvm'][:3]}, realised top-half accuracy "
              f"QSVM {q['qsvm_top_half_accuracy_realised']:.3f} / RBF {q['rbf_top_half_accuracy_realised']:.3f} "
              f"({time.time() - t0:.1f} s)")

        t0 = time.time()
        b = payload(await client.call_tool("brew_portfolio", {"coffee": "latte", "k": 3, "signal": "prophet",
                                                               "assets": SECTORS, "as_of": "2026-09-30"}))
        assert len(b["portfolio"]) == 3 and abs(sum(b["portfolio"].values()) - 1) < 1e-12, b
        print(f"brew latte    9 qubits, Prophet signal: QAOA {sorted(b['portfolio'])} exact {sorted(b['exact_optimum'])} "
              f"P(opt) {b['p_optimum_in_qaoa_state']:.3f} vs uniform {b['p_optimum_uniform_valid']:.3f} ({time.time() - t0:.1f} s)")

        t0 = time.time()  # default as_of = last day of the frozen file: Prophet refits on stdio
        q = payload(await client.call_tool("qsvm_rank", {}))
        assert q["as_of"] == "2026-10-16" and len(q["ranking"]) == 18, q
        print(f"qsvm_rank {q['as_of']} (18 Prophet refits): top-3 {q['predicted_top_half_qsvm'][:3]} ({time.time() - t0:.1f} s)")

        vols = {}
        for coffee in ("espresso", "decaf"):  # 18 qubits each
            t0 = time.time()
            b = payload(await client.call_tool("brew_portfolio", {"coffee": coffee, "signal": "momentum",
                                                                   "as_of": "2026-09-30"}))
            assert len(b["portfolio"]) == 4, b
            vols[coffee] = b["ex_ante_volatility"]["exact_optimum"]
            print(f"brew {coffee:8s} 18 qubits, momentum: QAOA {sorted(b['portfolio'])} exact {sorted(b['exact_optimum'])} "
                  f"exact-optimum vol {vols[coffee]:.3f} ({time.time() - t0:.1f} s)")
        assert vols["decaf"] <= vols["espresso"] + 1e-12, vols  # more risk aversion never raises the optimum's variance

        t0 = time.time()
        b = payload(await client.call_tool("brew_portfolio", {"coffee": "americano", "as_of": "2026-09-30"}))
        print(f"brew americano 18 qubits, QSVM signal: QAOA {sorted(b['portfolio'])} exact {sorted(b['exact_optimum'])} "
              f"found optimum {b['qaoa_found_exact_optimum']} ({time.time() - t0:.1f} s)")

        bad = await client.call_tool("brew_portfolio", {"coffee": "frappuccino"})
        assert bad.is_error and "unknown coffee" in bad.content[0].text, bad
        print("unknown coffee rejected:", bad.content[0].text)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
