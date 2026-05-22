<div align="center">

<!-- Logo/Banner placeholder - uncomment and add your image -->
<!-- <img src="assets/banner.png" alt="HyperAgents Banner" width="800"> -->

<h1>HyperAgents</h1>

<p>Self-referential self-improving agents that can optimize for any computable task</p>

<p>
<a href="LICENSE.md"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="License: CC BY-NC-SA 4.0"></a>
<a href="https://arxiv.org/abs/2603.19461"><img src="https://img.shields.io/badge/arXiv-2603.19461-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<a href="https://ai.meta.com/research/publications/hyperagents/"><img src="https://img.shields.io/badge/-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
<a href="https://x.com/jennyzhangzt/status/2036099935083618487"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

---

</div>

> [!NOTE]
> **This is a research fork.** Upstream: [facebookresearch/Hyperagents](https://github.com/facebookresearch/Hyperagents).
> This fork adds a new domain, [**`swe_gym`**](domains/swe_gym/), that wraps [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) v2.3.0 as the per-task inner agent for SWE-Gym evaluation, plus a Hyperagents-style mutation surface ([`domains/swe_gym/seed_harness/`](domains/swe_gym/seed_harness/)) for the **Phase H** harness-only-evolve baseline. See the SWE-Gym extension section below.

## Setup
```bash
# API keys, put these into .env file
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

```bash
# Install things
sudo dnf install -y python3.12-devel
sudo dnf install -y graphviz graphviz-devel cmake ninja-build bzip2-devel zlib-devel ncurses-devel libffi-devel
```

```bash
# Create virtual environment
python3.12 -m venv venv_nat
source venv_nat/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
# To build the docker container
docker build --network=host -t hyperagents .
```

```bash
# Setup initial agents
bash ./setup_initial.sh
```

## Running HyperAgents

```bash
# See the script for args, and baseline selections
python generate_loop.py --domains <domain>
```

By default, outputs will be saved in `outputs/` directory.

## File Structure
- `agent/` code for using foundation models
- `analysis/` scripts used for plotting and analysis
- `domains/` code for each domain — including [`swe_gym/`](domains/swe_gym/) added by this fork
- `utils/` common code used in the repo
- `run_meta_agent.py` script to help run the meta agent and get the diffs
- `meta_agent.py` main implementation of the meta agent
- `task_agent.py` main implementation of the task agent
- `generate_loop.py` entry point for running the algorithm (patched in this fork to dispatch the `swe_gym` domain)

## Logs from Experiments

The experiment logs can be downloaded here: https://drive.google.com/drive/folders/164fKQWgLM18foOzSnpv0F_I3TNpX8u8-?usp=sharing

## SWE-Gym extension (this fork)

This fork adds a `swe_gym` domain that lets the Hyperagents outer loop optimize
an agent for SWE-Gym tasks (a SWE-bench-style benchmark whose train set was
explicitly de-duplicated against SWE-bench's test sets).

**Architecture — Path C of the project plan.**
- Hyperagents owns the outer loop: archive, parent selection, generation
  scheduling, and scoring bookkeeping (via the new `run_harness_swe_gym(...)`
  in `generate_loop.py`).
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (v2.3.0) owns
  the per-task inner agent loop (`DefaultAgent` + `DockerEnvironment` +
  `LitellmTextbasedModel`), running Qwen3.5-4B served by vLLM on
  `localhost:8001`.

### Mutation surface (Phase H — harness-only evolve)

The meta agent edits four files under
[`domains/swe_gym/seed_harness/`](domains/swe_gym/seed_harness/) (621 LOC
total, copied verbatim from mini-swe-agent):

| File | Lines | Source |
|---|---:|---|
| `agent.py` | 171 | mini-swe-agent `agents/default.py` |
| `model.py` | 45 | mini-swe-agent `models/litellm_textbased_model.py` |
| `environment.py` | 161 | mini-swe-agent `environments/docker.py` |
| `prompts.yaml` | 244 | mini-swe-agent `config/benchmarks/swebench_backticks.yaml` |

Stable utilities (exceptions, serialize) are intentionally NOT copied — the
seed harness still imports those from the installed `minisweagent` package so
the meta agent's attention lands on the interesting agent logic only.

### Quick start

```bash
# 1) Serve Qwen3.5-4B via vLLM (128 K context). The helper kills any prior
#    instance and waits until /v1/models returns 200.
meta/scripts/serve_qwen35.sh

# 2) Phase 0 baseline — batch-eval the seed harness on SWE-Gym-Lite[0:50]
python -m domains.swe_gym.harness  --output_dir ./outputs/initial_swe_gym_0 \
                                   --run_id initial_swe_gym_0 --slice 0:50
python -m domains.swe_gym.report   --output_dir ./outputs/initial_swe_gym_0

# 3) Phase H — harness-only evolve (no weight training)
meta/scripts/run_phase_h.sh --iters 5
```

The `meta/scripts/` directory and `meta/phase0/` evaluator live in the parent
project alongside this fork (see the project's mini-swe-agent root).

### Docker images

SWE-Gym uses the `xingyaoww/sweb.eval.x86_64.<id>:latest` pre-built images
(note the `_s_` separator, vs. swebench's `_1776_`). The mapping is in
`domains/swe_gym/utils.docker_image_for()`.

Scoring uses a custom evaluator that applies `test_patch` + `model_patch` in
the container and runs `FAIL_TO_PASS` / `PASS_TO_PASS` via pytest. This
bypasses swebench's `MAP_REPO_VERSION_TO_SPECS`, which does not cover SWE-Gym
repos like `getmoto/moto`.

### Caveats

- This fork is **CC BY-NC-SA 4.0** (inherited). Derivative works must stay
  under the same license; commercial use is restricted.
- The mini-swe-agent source under `seed_harness/` is MIT-licensed in its
  original repository. It is included here verbatim as the meta agent's
  mutation target — subsequent diffs become candidate variants in the
  archive.

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Citing
If you find this project useful, please consider citing:
```bibtex
@misc{zhang2026hyperagents,
      title={Hyperagents}, 
      author={Jenny Zhang and Bingchen Zhao and Wannan Yang and Jakob Foerster and Jeff Clune and Minqi Jiang and Sam Devlin and Tatiana Shavrina},
      year={2026},
      eprint={2603.19461},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.19461}, 
}
```

