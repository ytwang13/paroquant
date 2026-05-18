# Research-repo refactor guide (domain-agnostic)

A checklist for restructuring **any** research or production-ML codebase around clear boundaries: **define behavior in a library**, **run jobs via thin CLIs**, **reproduce via pinned shells**, **deploy via a separate runtime** (optional).

Inspired by layered ML repos (e.g. quantization pipelines); **not tied to ParoQuant, PyTorch, or papers.**

---

## 1. The organizing principle

Split the repo by **question**, not by file type:

| Question | Where it lives | Must not depend on |
|----------|----------------|-------------------|
| What is the method? | **Core package** (`<pkg>/`) | CLIs, shells, notebooks, serving |
| How do I run it once? | **Entry CLI** (`train.py`, `run.py`) | Experiment scripts |
| How do I ship it? | **Runtime / deploy** (optional) | Training-only hacks |
| How do I reproduce the paper / report? | **`experiments/`** | Duplicated algorithm code |

**Rule:** If changing it would change **scientific claims** or **product behavior**, it belongs in the core package. If it only changes **hyperparameters, seeds, or plots**, it belongs in `experiments/`.

---

## 2. Generic layout

```
project/
├── <pkg>/                    # importable; no argparse, no shell assumptions
│   ├── core.py               # main domain object(s) (model, layer, pipeline step)
│   ├── transforms.py         # pure logic / math / data transforms
│   ├── train.py              # optimization loops, losses, schedulers
│   ├── adapters.py           # internal representation ↔ external formats
│   └── util.py               # shared boring helpers (I/O, logging, caches)
├── native/                   # optional: C++/CUDA/Rust extensions
├── train.py                  # thin CLI: parse config → call <pkg>
├── scripts/                  # one job per script (export, eval, demo)
│   ├── export_dev.py         # “looks like prod but still easy to debug”
│   ├── export_prod.py        # production artifact
│   └── eval_*.py
├── runtime/                  # optional: serving, fused ops, framework plugins
├── experiments/              # shells + configs only; calls train.py / scripts/
├── pyproject.toml / requirements.txt
└── README.md                 # happy path for new users
```

Rename freely (`<pkg>`, `runtime`, `native`). Keep the **roles**.

---

## 3. Layer responsibilities (fill in for your domain)

| Layer | Generic responsibility | Your project (fill in) |
|-------|------------------------|-------------------------|
| `core.py` | Single place that defines “what happens on one unit of work” | e.g. one batch, one layer, one graph node |
| `transforms.py` | Stateless or lightly stateful operations | e.g. augment, quantize, rotate, featurize |
| `train.py` | Fit parameters to data; staged schedules | e.g. `fit_unit(...)`, `fit_epoch(...)` |
| `adapters.py` | Convert checkpoints ↔ ONNX ↔ HF ↔ protobuf | e.g. packed tensors, API schemas |
| `util.py` | Discovery, patching trees, datasets, device/cache | e.g. `get_submodules`, `load_data` |
| `train.py` (root) | Parse config, write `args.json`, invoke library | |
| `scripts/` | Side effects: save files, call APIs, print metrics | |
| `runtime/` | Fast path used in production only | e.g. server, mobile, batch job |
| `experiments/` | Pin commands for tables and figures | |

---

## 4. Core patterns (apply anywhere)

### 4.1 One object owns the full “effective” behavior

**Pattern:** One class/module is the **source of truth** for what downstream code should assume.

- **Training path:** `forward(...)` uses internal state.
- **Inspection path:** `effective_state()` (or `materialize()`, `render()`, `compile()`) returns what the rest of the system should treat as output **without** re-implementing steps elsewhere.

**Refactor:** Find N places that repeat the same pipeline steps; collapse into one type with `forward` + `effective_state`.

```python
class CoreUnit:
    def forward(self, x): ...
    def effective_state(self): ...      # canonical snapshot for export/debug
    def set_trainable(self, **groups: bool): ...
    def parameters_named(self, group: str): ...
    @classmethod
    def from_checkpoint(cls, state) -> "CoreUnit": ...
```

### 4.2 Staged training via named parameter groups

**Pattern:** Multi-phase fitting (warmup structure → fine-tune weights → calibrate heads) driven by **config**, not `if epoch > 10` scattered in code.

- Config lists stages: `{group: lr, ...}` per stage.
- Each stage: `set_trainable(**groups)` → build optimizer → `fit_unit(...)`.

**Refactor:** Replace copy-pasted training blocks with one `fit_unit` and declarative stage lists.

### 4.3 Blocked / sequential processing (when memory or causality requires it)

**Pattern:** Process the system in **units** (layers, shards, time windows, graph partitions):

1. Capture inputs to unit 0 once.
2. For each unit: optimize/fix unit while holding downstream targets fixed; propagate new activations/state to the next unit.
3. Checkpoint **per unit**, not whole system every time.

**Refactor:** Extract `list_units()`, `get_children(type)`, `replace_child()`, `capture_inputs()`, `shard_cache()`.

Use when: GPU/RAM limits, autoregressive depth, or pipeline parallelism—not only for neural nets.

### 4.4 Internal representation ≠ external representation

**Pattern:** `adapters.py` owns all conversions:

- `to_runtime_format(internal)` 
- `from_checkpoint(blob)`

**Refactor:** Ban format logic in `scripts/` and `runtime/`; one module, round-trip test optional but valuable.

### 4.5 Two export paths: dev/simulation vs production

| Path | Purpose | Typical consumer |
|------|---------|------------------|
| **Dev / simulation** | Same interfaces as prod ecosystem, but easy to inspect (FP32, JSON, mock) | eval harnesses, ablations, CI |
| **Production** | Optimized types, fused ops, custom config flags | server, edge, batch jobs |

**Refactor:**

- `scripts/export_dev.py` — load unit checkpoints → `effective_state()` → write into **standard** APIs your benchmarks already use.
- `scripts/export_prod.py` — load checkpoints → `runtime` modules → save deploy bundle.

Training must not import serving frameworks.

### 4.6 Native code as optional sibling package

**Pattern:** Hot loops in `native/` (or `crates/`, `csrc/`), thin Python/Rust/JS bindings, used only from `core.py` or `transforms.py`.

**Refactor:** Profile first; extract when the op is in the inner loop **and** needs autograd/custom semantics.

### 4.7 Small, resumable artifacts

**Pattern:**

- Files named `{unit_id}.{component_id}.<ext>` (not one giant checkpoint per step).
- `args.json` (or `config.yaml`) next to output dir.
- `--resume` skips units whose artifact exists.

**Refactor:** Enables crash recovery, partial ablations, and parallel workers per unit.

---

## 5. Configuration and entrypoints

### 5.1 One config type per major CLI

Use a structured config (dataclass, pydantic, tyro, simple_parsing, hydra—pick one):

- Self-documenting `--help`
- Serialize to `args.json` / `config.yaml` on every run
- Lists for multi-stage options (`--stages a b`, `--epochs 10 20`)

**Refactor:** Merge 15 argparse scripts into: **one trainer CLI** + **small single-purpose scripts** in `scripts/`.

### 5.2 Shell scripts only pin reproduction

```bash
export PYTHONPATH="$(pwd)"
python train.py --config experiments/configs/baseline.yaml --resume
```

No algorithm logic in `.sh` files.

---

## 6. `util.py`: the “boring” layer

Move here anything used **≥3 times** that is not domain science:

| Category | Examples |
|----------|----------|
| Loading | models, configs, datasets, credentials |
| Tree surgery | find/replace submodules, named children |
| Caching | CPU/GPU shard iterators, disk spill |
| Logging | tqdm-safe, structured logs |
| Environment | device selection, `empty_cache`, seeds |

**Test:** If deleting `util.py` breaks both `train.py` and `scripts/`, it belongs there.

---

## 7. Optional `runtime/` package

Add only when you need a **different dependency graph** than training:

- Fused implementations
- Framework plugins (TensorRT, vLLM, Spark UDF, etc.)
- Long-lived servers

Structure:

```
runtime/
├── adapters/          # import from <pkg>.adapters when possible
├── executor/          # fast implementations
└── backends/          # factory: create_client(backend, model, ...)
```

**Pattern:** `create_backend(name, **opts)` so `scripts/demo.py` stays ~50 lines.

---

## 8. Migration plan (any repo)

### Phase 0 — Inventory
- [ ] List all entrypoints (notebooks, shells, `main`).
- [ ] List every place that mutates “the thing you care about” (weights, rules, graph, index).
- [ ] Pick one reference instance (one model, one dataset, one service).

### Phase 1 — Extract `<pkg>/`
- [ ] `core.py` — single owner of effective behavior
- [ ] `transforms.py` — pure steps
- [ ] `train.py` — `fit_unit` / `fit_epoch`
- [ ] `util.py` — shared I/O and discovery

### Phase 2 — One trainer CLI + artifacts
- [ ] Root `train.py` + saved config snapshot
- [ ] Per-unit checkpoints + `--resume`

### Phase 3 — Export split
- [ ] `export_dev` (simulation / standard eval)
- [ ] `export_prod` (if applicable)

### Phase 4 — Native / runtime (if needed)
- [ ] Profile → extract hot path
- [ ] `runtime/` behind `export_prod` only

### Phase 5 — Reproduction hygiene
- [ ] Move paper/benchmark commands to `experiments/`
- [ ] Delete duplicated logic from old paths

### Phase 6 — Contracts
- [ ] Round-trip: checkpoint → `effective_state()` → export_dev → metric unchanged
- [ ] Document supported variants in one registry (models, datasets, envs)

---

## 9. Anti-patterns → fixes

| Anti-pattern | Fix |
|--------------|-----|
| Monolithic `train.py` (1000+ lines) | Root CLI + `<pkg>/train.py` |
| Notebooks are the “real” code | Notebooks call `<pkg>`; library is canonical |
| Full snapshot every iteration | Per-unit artifacts |
| Serving imports training | `runtime/` + `adapters.py` |
| Hyperparameters only in Slack/wiki | `args.json` + versioned experiment configs |
| Ablations = copy-paste repo | `experiments/` changes flags only |
| Same pipeline in train and eval | `effective_state()` single definition |
| Magic globals (`DEVICE`, paths) | `util` factories; inject in CLI |

---

## 10. Minimal stable interfaces (language-agnostic)

Your refactor is “done enough” when these exist:

```
CoreUnit
  forward(input) → output
  effective_state() → serializable snapshot
  set_trainable(groups...)
  parameters_named(group) → params
  from_checkpoint(state) → CoreUnit

fit_unit(unit, train_data, val_data, optim_spec, n_iter) → void

adapters
  to_runtime_format(internal) → external
  from_checkpoint(blob) → internal

scripts/export_dev
  for each unit: load checkpoint → effective_state → write standard format

scripts/export_prod  (optional)
  for each unit: load checkpoint → runtime unit → write deploy bundle
```

Everything else (shells, plots, servers) plugs into these five surfaces.

---

## 11. README split

| Doc | Contents |
|-----|----------|
| **Root README** | install, quickstart, train → export → run, links to artifacts |
| **experiments/README** | exact commands for figures/tables, seeds, extra envs, baselines |

Users should never need `experiments/` for a first successful run.

---

## 12. When to skip parts

| Skip | If |
|------|-----|
| `native/` | Profiling shows Python/JS is fine |
| `runtime/` | Dev export is enough (batch offline, no SLA) |
| Blocked/unit processing | Full system fits in memory and trains end-to-end |
| `export_prod` | No separate deploy target |
| `experiments/` | Library-only OSS with no reproduction burden |

Keep **`<pkg>/` + thin CLI + `effective_state()`** even for small projects.

---

## 13. Quick mapping worksheet (copy per project)

```
Domain unit (what we optimize per checkpoint):  _______________
Standard eval interface (dev export target):   _______________
Production interface (prod export target):      _______________
Unit ID in artifact filenames:                 _______________
Config tool (dataclass / hydra / …):           _______________
Stages of training (group → lr):               _______________
```

Example mappings (do not treat as required):

| Domain | Unit | Dev export | Prod export |
|--------|------|------------|-------------|
| LLM quant | transformer block / linear | HF `nn.Linear` + FP16 | INT4 custom module |
| Recommender | embedding table shard | Parquet + sklearn | TorchServe |
| Compiler pass | function / basic block | LLVM IR text | bitcode |
| Data pipeline | stage | CSV preview | Iceberg table |

---

## 14. One-paragraph summary

Refactor any repo by **(1)** putting the method in an importable package with one “effective behavior” API, **(2)** driving training through a single config-backed CLI and small resumable unit checkpoints, **(3)** splitting dev/simulation export from production export, **(4)** optionally adding native code and a runtime package that training never imports, and **(5)** relegating reproduction to `experiments/` shells that only change flags. The contracts between layers are **`effective_state()`** and **`adapters`**—not shared scripts or copy-pasted pipelines.
