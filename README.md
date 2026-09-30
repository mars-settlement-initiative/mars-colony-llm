# Mission 2 — Introduce AI reasoning with Gemini

Folder 3 uses an online Gemini model to choose collector actions in the same
Mars world as Folders 1 and 2. Students investigate how individual decisions,
interactions and colony-level outcomes relate.

## Quick start: Gemini Free Tier

The default provider is **Gemini**, using **gemini-3.1-flash-lite**. Google
currently lists this stable model with free input and output tokens on Free Tier.

1. Open [Google AI Studio](https://aistudio.google.com/apikey).
2. Create an API key belonging to a project that shows **Free Tier**.
3. Keep that project on Free Tier; do not enable billing for this exercise.
4. From the student project root (containing all three folders), run:

```powershell
./.venv/Scripts/python.exe -m pip install -r mars_colony_3/requirements.txt
./mars_colony_3/gemini_setup.ps1
```

The launcher prompts for the key with hidden input and opens the dashboard
on port 8767. It does not save the key to disk. If PowerShell scripts are
restricted, use these commands directly; no policy changes are needed:

```powershell
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Gemini API key' -AsSecureString)).Password
./.venv/Scripts/python.exe mars_colony_3/run.py --mode gemini
```

Alternatively, set GEMINI_API_KEY in your IDE's run configuration and run
mars_colony_3/run.py. Gemini is the default when --mode is omitted.
GOOGLE_API_KEY is accepted as a fallback; GEMINI_API_KEY takes precedence.
Never put keys in source files or prompts. .env files are not automatically loaded.

**Free Tier is a Google project setting, not an API request parameter.** A Paid
Tier project's key can incur charges for the same API/model. The code cannot
inspect or override your project's billing tier. It does not enable billing,
switch providers, or retry quota failures. Use a Free Tier key and check its
active quotas in AI Studio.

## Run and inspect

Short live batch experiment with saved logs:

```powershell
./.venv/Scripts/python.exe mars_colony_3/run.py --batch --mode gemini --steps 50 --max-calls 10
```

Offline demonstration requiring no credentials:

```powershell
./.venv/Scripts/python.exe mars_colony_3/run.py --mode mock
./.venv/Scripts/python.exe mars_colony_3/run.py --batch --mode mock --seed 42
```

Mock mode uses the original coded rules through the new decision interface.
It is not an LLM. Unmodified Folders 1 and 2 currently have equivalent behavior,
and mock mode matches them across all five supplied benchmark seeds.

The dashboard shows the grid, agent table, mission status and original plots,
plus last actions, sources, API calls, continued choices and fallbacks.
Step advances a tick; Play runs continuously; Reset recreates the same world
and resets the local call budget. World settings remain fixed in this view.
Live requests can make a step take several seconds.

Batch runs create timestamped directories under mars_colony_3/results/:

- report.json: mission metrics, mode/model, prompt hash, cadence, calls and tokens.
- decisions.jsonl: observation, action, explanation, source and elapsed time per turn.
- timeseries.csv: original Mesa DataCollector time series.
- prompt.txt: exact prompt used.

Use --output to choose a destination. Reusing an explicit destination overwrites
its result files; the default creates a new directory.

## Request limits and decision cadence

Gemini defaults:

- **120 API attempts per simulation**, shared by all collectors.
- **1 second between a completed request and the next request**.
- Up to **10 simulation ticks per selected journey** before reconsideration.
- **1,024 output tokens maximum** per response; token usage is recorded.

These are conservative teaching defaults, not Google's published quota.
Daily or token quotas can be exhausted before the local attempt ceiling.

COLLECT and RETURN_BASE choices can continue without another request.
Movement still occurs one cell per tick. Arrival, target depletion, or energy
at most distance-to-base + 2 triggers earlier reconsideration. This asks the
LLM again; it does not override the next valid decision. WAIT is not cached.
New deposits or changed peer intentions do not interrupt a valid journey before
reconsideration. A loaded collector returns automatically, as in the baseline.
Turns with only one available action need no API call.

On HTTP 429 (quota exhausted), further requests stop for the run and original
rules provide explicitly labelled fallbacks. Authentication, permission,
invalid-request and unavailable-model failures also stop calls. Temporary
timeouts, connection/service errors, truncated output, invalid JSON and invalid
generated decisions use fallback for that turn but allow later calls. All
fallbacks are counted and logged. There is no paid fallback.

An already-selected journey can continue within its cadence after the local
call ceiling is reached; the next new choice uses fallback. Provider failure
disables continuation as well.

Pacing uses wall-clock time and does not advance simulation time or consume
simulated resources. A run can take several minutes. Five seeds at 120 attempts
permit up to 600 requests. Restarting does not reset Google's quota. Pacing is
per run; concurrent simulations or students sharing a project share its quota.

### Configuration

```powershell
$env:GEMINI_MODEL = 'gemini-3.1-flash-lite'
$env:MARS_MAX_API_CALLS = '120'
$env:MARS_GEMINI_REQUEST_SECONDS = '1'
$env:MARS_GEMINI_DECISION_INTERVAL = '10'
```

Check active limits before increasing request frequency. An interval of 1 asks
for a fresh choice every eligible collector turn. Keep cadence constant when
comparing prompts; it is part of the experiment and is recorded. --model
overrides GEMINI_MODEL. Verify any alternative model's free-tier availability.

### Result labels

| Result label | Meaning |
| --- | --- |
| mock_rules | Offline coded rules; no LLM |
| llm | Valid LLM choices, continued journeys and forced actions |
| llm_with_fallback | Mixture of LLM and original rules |
| no_llm_decisions | Live mode selected, but no LLM choice was made |

Per-turn sources are llm, continued_llm, forced_action, fallback or mock_rules.
llm_decisions counts new valid responses; continued_llm_decisions counts turns
reusing a choice. Inspect these and fallback counts before attributing results.

## Code structure

| File | Responsibility |
| --- | --- |
| agents.py | Existing mechanics, explorers, collector observations/actions |
| model.py | Same world creation, activation and consumption as Folder 2 |
| config.py | Copied Folder 2 world and benchmark settings |
| llm_config.py | Provider, model, cadence, pacing and call ceiling |
| prompt.txt | **Student modification area:** collector strategy |
| llm_decision.py | Routing, validation, continuation, pacing, fallback and logs |
| gemini_client.py | Native Gemini REST requests and response/error handling |
| gemini_setup.ps1 | Secure key prompt and dashboard launcher |
| run.py / app.py | Batch experiments / interactive dashboard |
| analysis.py | Three-folder benchmark and paired comparisons |
| benchmark_worker.py | Runs each folder with isolated imports |
| metrics.py | Identical observational measurements across folders |

The flow is: collect/unload/recharge -> observe -> continue or request a
validated decision -> move one cell or wait -> advance the simulation.

The LLM controls empty collectors' energy, target-selection and waiting choices.
Explorers keep their original rules; loaded collectors return automatically.
This is a hybrid rule/LLM agent. It does not rewrite Python or control the colony.
baseline_step() remains as a readable reference.

Observations include own position, energy, cargo, capacity, base location,
known resources, remembered target and other active collectors' shared targets.
A remembered target can be depleted. Undiscovered positions and resource amounts
are withheld. Shared targets are equally accessible to Folder 2 student rules;
they are intentions, not reservations.

- COLLECT_x_y selects a known/remembered target and moves one cell toward it.
  Collection occurs through the original mechanics at the target on a later turn.
- RETURN_BASE moves one cell toward base.
- WAIT stays still and retains target memory.

Loaded collectors have only RETURN_BASE; empty collectors at base cannot choose
RETURN_BASE. Every decision is checked against the current allowed actions.
Valid LLM mistakes can affect outcomes; physical mechanics remain unchanged.

## Compare all three folders

From the student project root:

```powershell
./.venv/Scripts/python.exe mars_colony_3/analysis.py --mode mock
./.venv/Scripts/python.exe mars_colony_3/analysis.py --mode gemini --max-calls 20
```

Seeds 10, 20, 30, 40 and 50 run in each folder with a 500-tick maximum.
The benchmark loads each folder's actual current code, so Folder 2 student edits
appear in the comparison. World settings are checked before running. Separate
processes isolate identically named imports; source hashes/settings are recorded.

Outputs: benchmark.csv, benchmark.json, summary.json (means and sample standard
deviations), paired_differences.json (each strategy minus Folder 1 on the same
seed), and Folder 3 decision logs and prompt.

Common metric definitions:

- steps_survived: ticks executed, including the resource-depletion tick.
- resources_collected: resources delivered to base, excluding carried cargo.
- distance_travelled: original movement counter; also movement energy use.
- failed_robots: robots inactive at the end.
- idle_collector_steps: active-at-start collector turns without recorded movement.
- duplicate_target_steps: after each tick, extra active empty collectors sharing
  a still-known target (three targeting one deposit add two).

Duplicate targeting measures overlapping intentions, not new duplicate selections.
Probes read state without modifying it or consuming random numbers. Matched seeds
give matching initial worlds; later trajectories can differ when policies consume
different random draws. Live LLM outputs can vary. Use multiple runs.

## Student challenge

1. Follow a collector's observation and action in the offline demo.
2. Benchmark your human-designed rules from Folder 2.
3. State a hypothesis linking an individual decision to a system outcome.
4. Edit the strategy section of prompt.txt and start a fresh Gemini run.
5. Compare the same seeds; inspect LLM, continuation and fallback counts.
6. Explain **individual decision -> interaction pattern -> system outcome**.
   A model's short rationale is not proof of a causal explanation.

Keep world settings, mechanics and decision cadence fixed when comparing prompts.
If Folder 2 changes explorers as well, discuss that additional difference.

The existing world has 100 initial resources plus 12 deposits of 20, consumed at
one per tick. Thus 340 ticks is an upper bound even with perfect delivery:
surviving 500 is impossible with finite stocks. The high robot energy also makes
failures uncommon. Folder 3 preserves these settings for comparability.

## Optional OpenAI provider

--mode openai retains the previous integration. It needs OPENAI_API_KEY and uses
OPENAI_MODEL (default gpt-4o-mini), billed according to that account. Gemini never
selects it automatically. OpenAI retains every-turn decisions without Gemini
pacing, so its default decision cadence differs from Gemini.

## Verification and references

From this folder:

```powershell
../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Tests intercept HTTP requests and need no real credentials or paid calls.
They cover Gemini REST requests, the OpenAI SDK, invalid/blocked responses,
quotas, pacing, continuation, hidden observations, conservation, and exact
original/mock trajectories. Live API performance still needs measurement with
the chosen account and model.

Official Google references: [generateContent](https://ai.google.dev/api/generate-content),
[structured outputs](https://ai.google.dev/gemini-api/docs/structured-output),
[pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
and [billing tiers](https://ai.google.dev/gemini-api/docs/billing).
