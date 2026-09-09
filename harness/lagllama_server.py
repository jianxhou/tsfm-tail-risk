"""Persistent Lag-Llama worker (gate condition 1): loads the checkpoint ONCE and
serves many windows over a newline-delimited JSON protocol on stdin/stdout.

Runs INSIDE .venv-lag-llama. Protocol (one JSON object per line):
    in : {"mode":"params","ctx":[...],"levels":[...],"seed":int}
         {"mode":"samples","ctx":[...],"n":int,"seed":int}
         {"mode":"ping"}                     -> {"pong":true}
    out: {"quantiles":[...],"df":f,"loc":f,"scale":f,"affine_loc":f,"affine_scale":f}
         {"samples":[...]}
EOF on stdin -> clean exit. Predictors are cached by context length (one build
for a fixed-ctx grid). The analytic params path reuses the exact hook capture of
the official forward pass validated in lagllama_worker.py.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

HF_REPO = "time-series-foundation-models/Lag-Llama"
HF_REV = "72dcfc29da106acfe38250a60f4ae29d1e56a3d9"
CHUNK = 512


def _load():
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="lag-llama.ckpt", revision=HF_REV)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt_path, ckpt["hyper_parameters"]["model_kwargs"]


def _make(ckpt_path, args, ctx_len, n_par):
    from lag_llama.gluon.estimator import LagLlamaEstimator
    est = LagLlamaEstimator(
        ckpt_path=ckpt_path, prediction_length=1, context_length=ctx_len,
        input_size=args["input_size"], n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"], n_head=args["n_head"],
        scaling=args["scaling"], time_feat=args["time_feat"],
        batch_size=1, num_parallel_samples=n_par)
    module = est.create_lightning_module()
    predictor = est.create_predictor(est.create_transformation(), module)
    return module, predictor


def _dataset(ctx):
    from gluonts.dataset.pandas import PandasDataset
    df = pd.DataFrame({"target": ctx},
                      index=pd.date_range("2000-01-03", periods=len(ctx), freq="D"))
    return PandasDataset(df, target="target")


def main() -> None:
    ckpt_path, args = _load()
    from scipy import stats
    cache: dict = {}   # (ctx_len, n_par, mode) -> (module, predictor)
    sys.stderr.write("lagllama_server ready\n"); sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        mode = req.get("mode", "samples")
        if mode == "ping":
            print(json.dumps({"pong": True}), flush=True); continue

        ctx = np.asarray(req["ctx"], dtype=np.float32)
        seed = int(req.get("seed", 20260706))
        torch.manual_seed(seed); np.random.seed(seed)
        L = min(len(ctx), 1024)

        if mode == "params":
            key = (L, 1, "params")
            if key not in cache:
                cache[key] = _make(ckpt_path, args, L, 1)
            module, predictor = cache[key]
            captured = {}
            orig = module.model.forward
            def hook(*a, __orig=orig, **kw):
                out = __orig(*a, **kw)
                captured.setdefault("out", out)
                return out
            module.model.forward = hook
            try:
                next(iter(predictor.predict(_dataset(ctx))))
            finally:
                module.model.forward = orig
            params, loc, scale = captured["out"]
            df_t, loc_t, scale_t = [float(p[0, -1]) for p in params]
            aff_l = float(np.asarray(loc.detach()).reshape(-1)[0])
            aff_s = float(np.asarray(scale.detach()).reshape(-1)[0])
            levels = [float(x) for x in req["levels"]]
            q = [aff_l + aff_s * (loc_t + scale_t * float(stats.t.ppf(a, df_t)))
                 for a in levels]
            print(json.dumps({"quantiles": q, "df": df_t, "loc": loc_t,
                              "scale": scale_t, "affine_loc": aff_l,
                              "affine_scale": aff_s}), flush=True)
        else:
            n = int(req["n"])
            key = (L, min(n, CHUNK), "samples")
            if key not in cache:
                cache[key] = _make(ckpt_path, args, L, min(n, CHUNK))
            _, predictor = cache[key]
            out: list = []
            ci = 0
            ds = _dataset(ctx)
            while len(out) < n:
                torch.manual_seed(seed + ci)
                fc = next(iter(predictor.predict(ds)))
                out.extend(fc.samples[:, 0].astype(float).tolist())
                ci += 1
            print(json.dumps({"samples": out[:n]}), flush=True)


if __name__ == "__main__":
    main()
