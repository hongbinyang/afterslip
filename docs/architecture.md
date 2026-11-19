# System architecture

Status: draft. This is the big picture of the intended Afterslip system: its components, how they depend on each other, and how data and control flow through them. It describes the target architecture; most components do not exist yet, and the status column says which do. Interfaces and detailed decisions are in [design.md](design.md); work items are in [milestones.md](milestones.md); the [requirements](requirements.md) remain authoritative.

## 1. Overview

Afterslip takes a pretrained robot policy (π₀), lets it practise recovering from injected drops and misgrasps with reinforcement learning, and measures two things for every policy: how often it recovers, and how much of its original skill it keeps. The primary disturbed comparison starts every policy from the **same** saved disturbed states; a secondary suite injects disturbances during each policy's own rollouts.

```mermaid
flowchart LR
    A["Pretrained policy π₀"] --> B["Disturbances:<br/>drops, misgrasps"]
    B --> C["RL fine-tuning<br/>without and with a<br/>forgetting mitigation"]
    C --> D["Evaluation:<br/>same saved states for every policy,<br/>plus online disturbances"]
    D --> E["Verdict:<br/>recovery vs retention"]
```

The system has **12 components in five layers**, plus a run record shared by all of them.

| Layer | Component | Purpose | Status (2026-11-19) |
|---|---|---|---|
| A. Foundation | 1. Environment and tooling | Locked environment, Makefile, tests, CI | Partly exists: locks and Makefile validated on macOS only; Linux install and CI not yet validated (M0 0.0c) |
| | 2. Artifact registry | Verified checkpoints, demonstrations, robot models, with hashes and licenses | Planned (M0 0.1, 0.2) |
| B. Simulation | 3. Task interface and adapters | The only door to the simulator; Core (robosuite Can) and FR3 (MuJoCo) adapters | Core adapter exists (M0 D1); FR3 adapter in F0 |
| | 4. Snapshot | Save and restore the full environment state | In-memory snapshots on one backend exist (M0 D2); serialization and hashing in M2. A simulator-state-only diagnostic fails as expected |
| | 5. Observer recorder | Videos from one fixed camera, with metadata | Exists (M0 D1) |
| C. Experiment building blocks | 6. Controllers | Fixed reference controller (scripted or π₀, chosen at M0) that reaches disturbance triggers; recovery controller that labels states | Planned (M0 0.5a, M2) |
| | 7. Disturbances | Triggers (when) and operators (what): drop, misgrasp, knock | Planned (M2) |
| | 8. Intervention-state bank | Generated disturbed states, recoverability labels, train/validation/test splits, hashes, test-access guard | Planned (M2) |
| D. Learning | 9. Policies | π₀ adapter, fine-tuned policies, toy policies; reset/reconstruction rule | Planned (M0, M1, M2) |
| | 10. RL trainer | ReinFlow wrapper, nominal/disturbed mix sampler, mitigation hooks | Planned (M1, M2) |
| E. Measurement | 11. Evaluation harness | Nominal, bank-based disturbed and end-to-end suites; per-episode records | Planned (M0, M2, M3) |
| | 12. Statistics and reporting | Hierarchical bootstrap, verdicts, headline figure, reports | Planned (M2, M3, M5) |
| (shared) | Run record | Config, commits, artifact hashes, runtime versions, outputs | Planned (M1) |

## 2. Component diagram

Arrows point from a component to what it uses.

```mermaid
flowchart TB
    subgraph A["A. Foundation"]
        ENV["1 Environment and tooling<br/>locks, Makefile, CI"]
        ART["2 Artifact registry<br/>checkpoint, demos, models"]
    end
    subgraph B["B. Simulation"]
        TASK["3 Task interface<br/>Can adapter, FR3 adapter"]
        SNAP["4 Snapshot"]
        REC["5 Observer recorder"]
    end
    subgraph C["C. Experiment building blocks"]
        CTRL["6 Controllers<br/>reference, recovery"]
        DIST["7 Disturbances<br/>triggers, operators"]
        BANK["8 Intervention-state bank<br/>splits, labels, guard"]
    end
    subgraph D["D. Learning"]
        POL["9 Policies<br/>π₀ adapter, fine-tuned"]
        RL["10 RL trainer<br/>mix sampler, mitigations"]
    end
    subgraph E["E. Measurement"]
        EVAL["11 Evaluation harness"]
        STATS["12 Statistics and reporting"]
    end
    RUN[("Run record<br/>outputs/")]

    SNAP --> TASK
    REC --> TASK
    CTRL --> TASK
    DIST --> TASK
    BANK --> CTRL
    BANK --> DIST
    BANK --> SNAP
    POL --> ART
    RL --> POL
    RL --> TASK
    RL --> BANK
    RL --> DIST
    EVAL --> POL
    EVAL --> TASK
    EVAL --> BANK
    EVAL --> DIST
    EVAL --> REC
    STATS --> EVAL
    RL --> RUN
    EVAL --> RUN
    STATS --> RUN
```

Design rules visible in the diagram:
- Everything that touches the simulator goes through the **task interface** (3). Swapping the robot is intended to mean a new adapter, not changes elsewhere.
- For the **primary disturbed suite**, the **bank** (8) supplies the starting states, so every policy is tested from identical physical and task states. The **end-to-end suite** (FR-EVAL-08) instead injects disturbances online during each policy's own rollout from nominal test seeds; it is reported separately with its trigger rate. The evaluation harness therefore uses both the bank and the disturbance module.
- During training, disturbed episodes come either from the **train** bank or from online injection (FR-DIS-08, decided in M2), so the trainer can use both.
- **Statistics** (12) read only recorded episodes, never the simulator, so every figure can be regenerated from logs.

## 3. Data flow: from artifacts to verdicts

What is produced, in order. Boxes are artifacts (files); labels on arrows say what produces them.

```mermaid
flowchart LR
    CK[("π₀ checkpoint<br/>+ hash")]
    DEMO[("Verified demonstrations")]
    REF["Fixed reference controller<br/>scripted or π₀"]
    BTV[("Bank: train + validation<br/>+ labels + hashes")]
    SEL[("Validation selection log<br/>M-none, chosen mitigation,<br/>baselines")]
    LOCK{{"Design-lock record<br/>committed before test bank"}}
    BT[("Bank: test<br/>+ hash")]
    FT[("Final checkpoints<br/>per configuration and seed")]
    EP[("Episode records + videos<br/>nominal, bank-based disturbed,<br/>end-to-end disturbed")]
    FIG[("Tables, headline figure,<br/>verdicts, report")]

    DEMO -- "pretraining, Path 2 only" --> CK
    CK -.-> REF
    REF -- "trigger, disturb, snapshot,<br/>label recoverability" --> BTV
    BTV -- "tuning on train/validation<br/>(or online injection)" --> SEL
    SEL --> LOCK
    LOCK -- "generated only after the lock" --> BT
    LOCK -- "final training,<br/>locked configs" --> FT
    FT -- "one test evaluation<br/>per configuration" --> EP
    CK --> EP
    BT --> EP
    EP -- "bootstrap, verdict rules" --> FIG
```

The dotted arrow marks the option that π₀ is the reference controller; the alternative is a scripted controller (chosen at M0, milestone 0.5a).

## 4. One bank-based disturbed evaluation episode

How the components interact when a policy is evaluated from one bank state. The same sequence runs for π₀ and for every final policy, from the same state.

```mermaid
sequenceDiagram
    participant H as Evaluation harness
    participant B as Bank
    participant T as Task interface
    participant P as Policy
    participant R as Observer recorder
    participant L as Episode records

    H->>B: load state k (bank-state ID, hash checked)
    B-->>H: snapshot + stored observation/action history
    H->>T: set_state(snapshot)
    T-->>H: observation
    Note over H,P: Reconstruction rule (FR-DIS-10)
    H->>P: reset() internal state
    H->>P: warm_start(history) rebuilds frame stack or recurrent state
    H->>P: discard any pending action chunk
    H->>R: start(episode metadata)
    loop each control step until success or horizon
        H->>P: act(observation), replanning from the restored state
        P-->>H: action
        H->>T: step(action)
        T-->>H: observation, success, timeout
        H->>R: capture frame
    end
    H->>R: finish() writes video + metadata
    H->>L: append record (success, steps, re-grasps, IDs, video path)
```

The reconstruction rule is a practical approximation; it does not guarantee identical hidden policy state across architectures. A policy type that cannot be initialised this way is reported as a limitation, and its disturbed results come from the end-to-end suite instead. The end-to-end suite itself runs a nominal episode from a test seed, injects the disturbance online when the trigger fires during the policy's own rollout, and records the trigger rate.

## 5. Training loop with disturbances and mitigation

How fine-tuning mixes normal and disturbed practice, and where a forgetting mitigation acts. The Core comparison always includes **M-none** (plain fine-tuning, no mitigation) and **one** chosen mitigation; the others are Extended. All hyperparameters, including the mitigation's, are selected on validation data before the design lock.

```mermaid
flowchart LR
    subgraph Sampler["Mix sampler (disturbed variants)"]
        NOM["Nominal start<br/>(seeded reset)"]
        DIS["Disturbed start<br/>(train bank or online injection)"]
    end
    NOM --> ROLL
    DIS --> ROLL
    ROLL["Rollouts<br/>policy × task interface"] --> UPD["ReinFlow update<br/>sparse success reward"]
    UPD --> POLICY["Policy being fine-tuned"]
    POLICY --> ROLL
    PI0["Frozen π₀"] -. "M-reg: stay close to π₀" .-> UPD
    DEMOS["Original demonstrations"] -. "M-aux: rehearse demos" .-> UPD
    FREEZE["Frozen lower layers"] -. "M-freeze: limit what changes" .-> POLICY
    UPD --> CKPT[("Checkpoints at fixed<br/>environment-step intervals")]
```

Dotted arrows are optional: **M-none uses none of them**; each mitigation variant uses exactly one. The nominal-only baseline uses only nominal starts.

## 6. The fair-test protocol

The order in which data may be used. This makes the final comparison trustworthy: nothing about the design is chosen by looking at test results. The requirement is that the design-lock record is committed **before** the test bank exists (FR-DIS-05); a git tag is the proposed mechanism for proving the order.

```mermaid
stateDiagram-v2
    [*] --> Develop
    Develop: Develop and tune (M1-M2)<br/>train + validation data only<br/>M-none, chosen mitigation, baselines
    Develop --> Locked: commit design-lock record
    Locked: Design locked<br/>all selected values fixed
    Locked --> TestBank: generate and hash test bank
    TestBank: Test bank exists<br/>never loaded by training or tuning code
    TestBank --> Evaluate: train locked configs
    Evaluate: One test evaluation<br/>per configuration (M3)
    Evaluate --> Verdicts: predeclared statistics
    Verdicts --> [*]
```

## 7. FR3 track: intended reuse

The FR3 track applies the same machinery to the Franka Research 3 arm in a three-bin sorting scene. **This is an architectural goal, not a guarantee:** each reused component must pass compatibility tests with the FR3 adapter before it is relied on, and changes are made where tests show they are needed.

```mermaid
flowchart TB
    subgraph Reused["Intended for reuse (subject to compatibility tests)"]
        R1["4 Snapshot"]
        R2["5 Observer recorder"]
        R3["7 Disturbance framework"]
        R4["8 Bank and splits"]
        R5["10 RL trainer + mitigations"]
        R6["11 Evaluation harness"]
        R7["12 Statistics and reporting"]
    end
    subgraph New["New for FR3"]
        N1["FR3 adapter<br/>plain MuJoCo"]
        N2["Sorting scene<br/>3 bins, objects, observer camera"]
        N3["Hand on the FR3 flange"]
        N4["FR3 controllers<br/>joint, end-effector"]
        N5["FR3 base policy<br/>path chosen in F0:<br/>scripted demos, retargeting,<br/>or existing checkpoint"]
    end
    N1 --> N2
    N1 --> N3
    N4 --> N1
    N5 --> N4
    Reused -- "talk only to the task interface" --> N1
```

The FR3 track has two phases. **F0** (alongside M0–M2) builds the adapter, hand and controllers step by step, each with a video, and ends with a decision on the F1 path. **F1** (after M3) builds the sorting scene and base policy, then runs the method in a scope decided at F1 entry: a **demonstration**, or a **research comparison**. A research comparison has its own train/validation/test data, its own validation-only selection and its own design lock (section 6, repeated for FR3); values locked for Core are not assumed to transfer.

## 8. Simplified view for the README

```mermaid
flowchart LR
    P["Pretrained<br/>robot policy"] --> S["Simulation<br/>(MuJoCo)"]
    S --> D["Inject drops<br/>and misgrasps"]
    D --> R["RL practice,<br/>with and without<br/>anti-forgetting"]
    R --> E["Same saved tests<br/>for every policy"]
    E --> V["Recovery vs<br/>retained skill"]
```

## 9. Animation (planned)

An animated SVG of the end-to-end flow (sections 3 and 4 combined), in the style of the explainer animations and generated by a script like `docs/explainer/make_svgs.py`, is planned once the pipeline runs end to end (after M2–M3), so that it shows the real system rather than a plan.
