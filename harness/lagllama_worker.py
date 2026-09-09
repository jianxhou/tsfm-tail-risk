"""Worker executed INSIDE .venv-lag-llama (dep-quarantined; CHANGELOG 2026-07-04).

Protocol: single JSON object on stdin -> single JSON object on stdout.
    in : {"mode": "samples", "ctx": [...], "n": int, "seed": int}
         {"mode": "params",  "ctx": [...], "levels": [floats], "seed": int}
    out: {"samples": [...]}                                  (samples mode)
         {"quantiles": [...], "df": f, "loc": f, "scale": f} (params mode —
          analytic Student-t quantiles from the distribution the OFFICIAL forward
          pass produces; captured via a forward hook, zero reimplementation)
Only stdlib + the venv's own packages here — this file must stay importable by
the isolated interpreter, so no harness/ imports. NB: the lag-llama pip package
ships a top-level `data` module; never run this worker with a cwd whose `data/`
package would shadow it (script mode puts this file's dir first, which is safe).
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

HF_REPO = "time-series-foundation-models/Lag-Llama"
HF_REV = "72dcfc29da106acfe38250a60f4ae29d1e56a3d9"   # registry pin


def main() -> None:
    req = json.loads(sys.stdin.read())
    ctx = np.asarray(req["ctx"], dtype=np.float32)
    mode = req.get("mode", "samples")
    n = int(req.get("n", 1)) if mode == "samples" else 1
    seed = int(req.get("seed", 20260704))
    torch.manual_seed(seed)
    np.random.seed(seed)

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="lag-llama.ckpt",
                                revision=HF_REV)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["hyper_parameters"]["model_kwargs"]

    from lag_llama.gluon.estimator import LagLlamaEstimator
    CHUNK = 512   # parallel-sample batch cap; larger n is drawn in seeded chunks
    def make_predictor(n_par: int):
        est = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=1,
            context_length=min(len(ctx), 1024),
            input_size=args["input_size"],
            n_layer=args["n_layer"],
            n_embd_per_head=args["n_embd_per_head"],
            n_head=args["n_head"],
            scaling=args["scaling"],
            time_feat=args["time_feat"],
            batch_size=1,
            num_parallel_samples=n_par,
        )
        module = est.create_lightning_module()
        transform = est.create_transformation()
        return module, est.create_predictor(transform, module)

    module, predictor = make_predictor(min(n, CHUNK) if mode == "samples" else 1)

    captured = {}
    if mode == "params":
        # capture (params, loc, scale) the OFFICIAL forward pass produces, so the
        # analytic quantile uses byte-identical inputs to the sampling path
        orig_fwd = module.model.forward

        def hook(*a, **kw):
            out = orig_fwd(*a, **kw)
            if "out" not in captured:
                captured["out"] = out
            return out

        module.model.forward = hook

    from gluonts.dataset.pandas import PandasDataset
    df = pd.DataFrame({"target": ctx},
                      index=pd.date_range("2000-01-03", periods=len(ctx), freq="D"))
    ds = PandasDataset(df, target="target")

    if mode == "samples":
        out: list[float] = []
        chunk_i = 0
        while len(out) < n:
            torch.manual_seed(seed + chunk_i)
            fc = next(iter(predictor.predict(ds)))
            out.extend(fc.samples[:, 0].astype(float).tolist())
            chunk_i += 1
        json.dump({"samples": out[:n]}, sys.stdout)
        return

    fc = next(iter(predictor.predict(ds)))

    params, loc, scale = captured["out"]
    df_t, loc_t, scale_t = [float(p[0, -1]) for p in params]   # StudentTOutput arg order
    aff_loc = float(np.asarray(loc.detach()).reshape(-1)[0])
    aff_scale = float(np.asarray(scale.detach()).reshape(-1)[0])
    from scipy import stats
    levels = [float(x) for x in req["levels"]]
    q = [aff_loc + aff_scale * (loc_t + scale_t * float(stats.t.ppf(a, df_t)))
         for a in levels]
    json.dump({"quantiles": q, "df": df_t, "loc": loc_t, "scale": scale_t,
               "affine_loc": aff_loc, "affine_scale": aff_scale}, sys.stdout)


if __name__ == "__main__":
    main()
