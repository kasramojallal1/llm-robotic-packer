"""
GOPT baseline (Xiong et al., RA-L 2024) behind the harness Policy interface (T10.1-T10.3; R1.10; D70-D73).

GOPT's code is NOT vendored (academic-use-only licence, no redistribution grant, D70).
Clone the official repository at the pinned commit and point GOPT_DIR at it:

    git clone https://github.com/Xiong5Heng/GOPT.git && git -C GOPT checkout a2e42de1c0ab62c5e0a356e363349c32beb9e05b
    export GOPT_DIR=$PWD/GOPT
    export GOPT_CKPT=models/gopt/policy_step_final.pth      # trained with GOPT's own ts_train.py (D70)

Only GOPT's own Placement Generator (envs/Packing/container.py + ems.py) and its
network (model.py) are loaded, straight from GOPT_DIR; nothing in them is modified.
gymnasium / tianshou / vtk are not needed at test time.

Protocol (what differs from GOPT's own test script, each one decided and documented):
  - GOPT sees the bin as its height map, built from the placed boxes (`observe`);
    one item at a time, no lookahead, no unpacking (= every other method).
  - Orientations: GOPT's two upright yaws, (l, w, h) and (w, l, h) of the item as it
    appears in the sequence file (D71).  Mapped onto the harness rotation_index.
  - Candidates: GOPT's EMS candidates (k = 80) with GOPT's own mask, intersected with
    the harness placement contract (harness/validator.validate_pick: bounds, collision,
    full-base support, vertical clearance) before the network chooses (D72).  The
    validator still checks the returned placement inside the runner's 3 x 2 budget.
    A box with no candidate left is skipped and the sequence continues (harness rule).
  - Action decoding: index // k = orientation, index % k = EMS index (the network's
    layout).  GOPT's env.idx2pos uses `idx >= k - 1`, which mislabels index k-1; the
    correct decoding is used here.
  - Path: GOPT produces none; the deterministic template path (T3.3) is used so the
    validator can run - packing is assessed separately from waypoint generation.
  - Action choice: `gopt` takes the highest-probability action (argmax); `gopt-sample`
    samples from the masked policy as GOPT's ts_test.py does (deterministic_eval is
    False there), with a torch generator seeded from the sequence seed.

Also here: the rotation control of D71, `gopt-control-greedy-2rot` = greedy (Eq. 5)
restricted to the same two upright orientations GOPT may use.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import types
from typing import Dict, List, Optional

import numpy as np

from harness.policies import GreedyPolicy, Policy, PolicyOutput, _repo_relative, _TemplatePathMixin
from harness.validator import validate_pick

GOPT_COMMIT = "a2e42de1c0ab62c5e0a356e363349c32beb9e05b"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def upright_rotation_indices(state: Dict) -> Dict[int, int]:
    """{yaw flag (0 = as given, 1 = x-y swapped): harness rotation_index} for the item's two upright yaws."""
    l, w, h = state["incoming_box"]["original_size"]
    rots = [list(r) for r in state["incoming_box"]["rotations"]]
    out = {}
    for flag, size in ((0, [l, w, h]), (1, [w, l, h])):
        if size in rots:
            out[flag] = rots.index(size)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gopt_modules(gopt_dir: str):
    """Import GOPT's container (Placement Generator) and model.py under private names.

    GOPT's package is called `envs`, like ours, so it is mounted as `gopt_packing`
    without running its __init__ (which imports gymnasium and the vtk renderer).
    """
    pkg_dir = os.path.join(gopt_dir, "envs", "Packing")
    if not os.path.isfile(os.path.join(pkg_dir, "container.py")):
        raise FileNotFoundError(f"GOPT_DIR={gopt_dir!r} is not a GOPT checkout (see harness/gopt_policy.py)")
    if "gopt_packing" not in sys.modules:
        pkg = types.ModuleType("gopt_packing")
        pkg.__path__ = [pkg_dir]
        sys.modules["gopt_packing"] = pkg
    container = importlib.import_module("gopt_packing.container")
    spec = importlib.util.spec_from_file_location("gopt_model", os.path.join(gopt_dir, "model.py"))
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return container, model


class GOPTPolicy(_TemplatePathMixin, Policy):
    free_placement = True       # returns {"rotation_index", "position"}; the runner validates it

    def __init__(self, name: str = "gopt", gopt_dir: Optional[str] = None, ckpt: Optional[str] = None,
                 sample: bool = False, device: Optional[str] = None):
        import torch
        import yaml

        self.name = name
        self.sample = sample
        self.deterministic = True   # argmax, or sampling from a generator seeded per sequence
        self.gopt_dir = os.path.abspath(gopt_dir or os.environ.get("GOPT_DIR", ""))
        ckpt = ckpt or os.environ.get("GOPT_CKPT") or os.path.join(_REPO_ROOT, "models", "gopt", "policy_step_final.pth")
        self.ckpt = os.path.abspath(ckpt)
        self._container_mod, model_mod = _load_gopt_modules(self.gopt_dir)

        # network hyper-parameters: the config.yaml ts_train.py copied next to the checkpoint, else GOPT's default
        cfg_path = os.path.join(os.path.dirname(self.ckpt), "config.yaml")
        if not os.path.isfile(cfg_path):
            cfg_path = os.path.join(self.gopt_dir, "cfg", "config.yaml")
        with open(cfg_path) as f:
            c = yaml.safe_load(f)
        env, mdl = c["env"], c["model"]
        self.cfg_path = cfg_path
        self.container_size = [int(v) for v in env["container_size"]]
        self.k = int(env["k_placement"])
        box_big = int(max(self.container_size) / 2)       # arguments.py
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = model_mod.ShareNet(k_placement=self.k, box_max_size=box_big, container_size=self.container_size,
                                 embed_size=mdl["embed_dim"], num_layers=mdl["num_layers"],
                                 forward_expansion=mdl["forward_expansion"], heads=mdl["heads"],
                                 dropout=mdl["dropout"], device=self.device, place_gen=env["scheme"])
        self.actor = model_mod.ActorHead(preprocess_net=net, embed_size=mdl["embed_dim"],
                                         padding_mask=mdl["padding_mask"], device=self.device).to(self.device)
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)                                   # checkpoint.pth = {"model", "optim"}
        actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor_sd, strict=True)
        self.actor.eval()
        self.ckpt_sha256 = _sha256(self.ckpt)
        self.model_id = f"GOPT@{GOPT_COMMIT[:7]}:{os.path.basename(self.ckpt)}:{self.ckpt_sha256[:12]}"
        self._gen = None
        self._box = None

    def describe(self) -> Dict:
        import torch
        return {"name": self.name, "model_id": self.model_id, "gopt_commit": GOPT_COMMIT,
                "checkpoint": _repo_relative(self.ckpt), "checkpoint_sha256": self.ckpt_sha256,
                "config": _repo_relative(self.cfg_path), "k_placement": self.k,
                "action": "sample" if self.sample else "argmax", "device": str(self.device),
                "torch": torch.__version__}

    # ---------------- episode / box hooks (harness/runner.py) ----------------

    def begin_episode(self, sequence: Dict):
        import torch
        if [int(v) for v in sequence["bin_dims"]] != self.container_size:
            raise ValueError(f"GOPT model trained for bin {self.container_size}, sequence has {sequence['bin_dims']}")
        self._gen = torch.Generator().manual_seed(int(sequence["seed"]))

    def observe(self, placed: List[Dict], size: List[int], bin_dims: List[int]):
        """Rebuild GOPT's bin from the placed boxes and compute its candidates + both masks for this box."""
        C = self._container_mod
        cont = C.Container(*self.container_size, rotation=True)
        for b in placed:
            (x, y, z), (sx, sy, sz) = b["position"], b["size"]
            cont.boxes.append(C.Box(sx, sy, sz, x, y, z))
            cont.heightmap = cont.update_heightmap(cont.heightmap, cont.boxes[-1])
        item = [int(v) for v in size]
        cands, gmask = cont.candidate_from_EMS(item, self.k)          # GOPT's Placement Generator, unchanged
        cand = np.zeros((self.k, 6), dtype=np.int32)
        if len(cands):
            cand[:len(cands)] = cands
        dims = {0: item, 1: [item[1], item[0], item[2]]}
        ours = np.zeros_like(gmask)
        targets = {}
        for flag in (0, 1):
            for i in range(min(len(cands), self.k)):
                if not gmask[flag, i]:
                    continue
                d = dims[flag]
                x, y = int(cand[i][0]), int(cand[i][1])
                z = int(np.max(cont.heightmap[x:x + d[0], y:y + d[1]]))   # gravity drop, as GOPT's place_box
                pos = [x, y, z]
                if validate_pick(pos, d, placed, bin_dims).ok:
                    ours[flag, i] = 1
                targets[flag * self.k + i] = (flag, pos, d)
        obs = np.concatenate([cont.heightmap.reshape(-1), np.array(item + [item[1], item[0], item[2]]),
                              cand.reshape(-1)]).astype(np.float32)
        self._box = {"obs": obs, "gopt_mask": gmask.reshape(-1).astype(bool), "mask": ours.reshape(-1).astype(bool),
                     "targets": targets, "tried": set(), "n_candidates": int(len(cands))}

    def has_placement(self, state: Dict) -> bool:
        return bool(self._box is not None and self._box["mask"].any())

    # ---------------- decision ----------------

    def _choose(self, mask: np.ndarray) -> int:
        import torch
        obs = types.SimpleNamespace(obs=self._box["obs"][None], mask=mask[None])
        with torch.no_grad():
            logits, _ = self.actor(obs)
        logits = logits[0].float().cpu()
        m = torch.as_tensor(mask)
        logits = torch.where(m, logits, torch.tensor(-1e18))
        if self.sample:
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1, generator=self._gen).item())
        return int(torch.argmax(logits).item())

    def pick(self, state, feedback):
        box = self._box
        mask = box["mask"].copy()
        for a in box["tried"]:
            mask[a] = False
        if not mask.any():
            return PolicyOutput(None, raw="gopt: no candidate left")
        a = self._choose(mask)
        box["tried"].add(a)
        flag, pos, d = box["targets"][a]
        rmap = upright_rotation_indices(state)
        info = {"action": a, "ems_index": a % self.k, "yaw_flag": flag, "n_candidates": box["n_candidates"],
                "n_gopt_mask": int(box["gopt_mask"].sum()), "n_filtered_mask": int(box["mask"].sum())}
        if len(box["tried"]) == 1:
            # D72 diagnostic: what GOPT would have chosen with only its own (looser) rule
            gen_state = self._gen.get_state() if self._gen is not None else None
            u = self._choose(box["gopt_mask"].copy())
            if gen_state is not None:
                self._gen.set_state(gen_state)
            info["unfiltered_choice"] = u
            info["unfiltered_choice_breaks_our_rules"] = bool(not box["mask"][u])
        return PolicyOutput({"rotation_index": rmap[flag], "position": pos}, raw=None, info=info)


# ------------------------ D71 control ------------------------

class Greedy2RotPolicy(GreedyPolicy):
    """Greedy (Eq. 5) over the offered anchors of the two upright yaws only (GOPT's rotation rule)."""

    def __init__(self, name: str):
        self.name = name

    def _allowed(self, state):
        keep = set(upright_rotation_indices(state).values())
        return [a for a in state["anchors_indexed"] if a["rotation_index"] in keep]

    def has_placement(self, state) -> bool:
        return bool(self._allowed(state))

    def pick(self, state, feedback):
        return super().pick(dict(state, anchors_indexed=self._allowed(state)), feedback)


def make_gopt_policy(spec: str, seed: int = 0) -> Policy:
    if spec == "gopt":
        return GOPTPolicy("gopt", sample=False)
    if spec == "gopt-sample":
        return GOPTPolicy("gopt-sample", sample=True)
    if spec == "gopt-control-greedy-2rot":
        return Greedy2RotPolicy(spec)
    raise KeyError(f"unknown GOPT method {spec!r}")
