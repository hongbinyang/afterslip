# System architecture

Status: draft. This is the big picture of the finished Afterslip system: its components, how they depend on each other, and how data and control flow through them. Interfaces and detailed decisions are in [design.md](design.md); work items are in [milestones.md](milestones.md). Components that do not exist yet are marked with the milestone that builds them.

## 1. Overview

Afterslip takes a pretrained robot policy (π₀), lets it practise recovering from injected drops and misgrasps with reinforcement learning, and measures two things on the **same** saved disturbances for every policy: how often it recovers, and how much of its original skill it keeps.

```mermaid
flowchart LR
    A["Pretrained policy π₀"] --> B["Disturbances:<br/>drops, misgrasps"]
    B --> C["RL fine-tuning<br/>+ forgetting mitigation"]
    C --> D["Fair evaluation:<br/>same saved states<br/>for every policy"]
    D --> E["Verdict:<br/>recovery vs retention"]
```

The system has **12 components in five layers**, plus a run record shared by all of them.

| Layer | Component | Purpose | Built in |
|---|---|---|---|
| A. Foundation | 1. Environment and tooling | Locked environment, Makefile, tests, CI | M0 (exists) |
| | 2. Artifact registry | Verified checkpoints, demonstrations, robot models, with hashes and licenses | M0 |
| B. Simulation | 3. Task interface and adapters | The only door to the simulator; Core (robosuite Can) and FR3 (MuJoCo) adapters | M0 (D1), F0 |
| | 4. Snapshot | Save and restore the full environment state | M0 (D2), M2 |
| | 5. Observer recorder | Videos from one fixed camera, with metadata | M0 (D1) |
| C. Experiment building blocks | 6. Controllers | Scripted reference controller (makes disturbances happen) and recovery controller (labels states) | M0, M2 |
| | 7. Disturbances | Triggers (when) and operators (what): drop, misgrasp, knock | M2 |
| | 8. Intervention-state bank | Generated disturbed states, recoverability labels, train/validation/test splits, hashes, test-access guard | M2 |
| D. Learning | 9. Policies | π₀ adapter, fine-tuned policies, toy policies; reset rule | M0, M1 |
| | 10. RL trainer | ReinFlow wrapper, nominal/disturbed mix sampler, mitigation hooks | M1, M2 |
| E. Measurement | 11. Evaluation harness | Nominal, disturbed and end-to-end suites; per-episode records | M0, M2, M3 |
| | 12. Statistics and reporting | Hierarchical bootstrap, verdicts, headline figure, reports | M2, M3, M5 |
| (shared) | Run record | Config, commit, artifact hashes, runtime versions, outputs | M1 |

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
    EVAL --> POL
    EVAL --> TASK
    EVAL --> BANK
    EVAL --> REC
    STATS --> EVAL
    RL --> RUN
    EVAL --> RUN
    STATS --> RUN
```

Design rules visible in the diagram:
- Everything that touches the simulator goes through the **task interface** (3). Swapping the robot means a new adapter, not changes elsewhere.
- The **bank** (8) is the only source of disturbed starting states for evaluation, so every policy is tested from identical states.
- **Statistics** (12) read only recorded episodes, never the simulator, so every figure can be regenerated from logs.

## 3. Data flow: from artifacts to verdicts

What is produced, in order. Boxes are artifacts (files); labels on arrows are the components that produce them.

```mermaid
flowchart LR
    CK[("π₀ checkpoint<br/>+ hash")]
    DEMO[("Demonstrations")]
    BTV[("Bank: train + validation<br/>+ labels + hashes")]
    LOCK{{"Design lock<br/>git tag"}}
    BT[("Bank: test<br/>+ hash")]
    FT[("Fine-tuned checkpoints<br/>per variant and seed")]
    EP[("Episode records<br/>+ videos")]
    FIG[("Tables, headline figure,<br/>verdicts, report")]

    DEMO -- "pretraining, Path 2 only" --> CK
    CK -- "controllers + disturbances<br/>(bank generator)" --> BTV
    BTV -- "validation tuning<br/>(RL trainer)" --> LOCK
    LOCK -- "generated only after the lock" --> BT
    CK -- "final training,<br/>locked configs" --> FT
    BTV --> FT
    FT -- "one test evaluation<br/>per configuration" --> EP
    CK --> EP
    BT --> EP
    EP -- "bootstrap, verdict rules" --> FIG
```

## 4. One disturbed evaluation episode

How the components interact when a policy is evaluated from one bank state. The same sequence runs for π₀ and for every fine-tuned policy, from the same state.

```mermaid
sequenceDiagram
    participant H as Evaluation harness
    participant B as Bank
    participant T as Task interface
    participant P as Policy
    participant R as Observer recorder
    participant L as Episode records

    H->>B: load state k (bank-state ID, hash checked)
    B-->>H: snapshot + observation/action history
    H->>T: set_state(snapshot)
    T-->>H: observation
    H->>P: reset(), warm_start(history)
    H->>R: start(episode metadata)
    loop each control step until success or horizon
        H->>P: act(observation)
        P-->>H: action
        H->>T: step(action)
        T-->>H: observation, success, timeout
        H->>R: capture frame
    end
    H->>R: finish() writes video + metadata
    H->>L: append record (success, steps, re-grasps, IDs, video path)
```

## 5. Training loop with disturbances and mitigation

How fine-tuning mixes normal and disturbed practice, and where the forgetting mitigation acts.

```mermaid
flowchart LR
    subgraph Sampler["Mix sampler"]
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

Only one mitigation is used in the Core comparison (chosen in M0); the others are Extended.

## 6. The fair-test protocol

The order in which data may be used. This is what makes the final comparison trustworthy: nothing about the design is chosen by looking at test results.

```mermaid
stateDiagram-v2
    [*] --> Develop
    Develop: Develop and tune (M1-M2)<br/>train + validation data only
    Develop --> Locked: commit design-lock record, tag
    Locked: Design locked<br/>all parameters fixed
    Locked --> TestBank: generate and hash test bank
    TestBank: Test bank exists<br/>never loaded by training code
    TestBank --> Evaluate: train locked configs
    Evaluate: One test evaluation<br/>per configuration (M3)
    Evaluate --> Verdicts: predeclared statistics
    Verdicts --> [*]
```

## 7. FR3 track: what is reused and what is new

The FR3 track applies the same machinery to the Franka Research 3 arm in a three-bin sorting scene. Only the robot- and task-specific parts are new.

```mermaid
flowchart TB
    subgraph Reused["Reused unchanged"]
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
        N5["FR3 base policy π₀<br/>scripted demos, retargeting,<br/>or existing checkpoint"]
    end
    N1 --> N2
    N1 --> N3
    N4 --> N1
    N5 --> N4
    Reused -- "talk only to the task interface" --> N1
```

The FR3 track proceeds in two phases: **F0** (alongside M0–M2) builds N1, N3 and N4 step by step, each with a video; **F1** (after M3) adds N2 and N5 and runs the method. An FR3 research comparison uses its own train/validation/test protocol and lock (section 6, repeated for FR3).

## 8. Simplified view for the README

```mermaid
flowchart LR
    P["Pretrained<br/>robot policy"] --> S["Simulation<br/>(MuJoCo)"]
    S --> D["Inject drops<br/>and misgrasps"]
    D --> R["RL practice<br/>+ anti-forgetting"]
    R --> E["Same tests<br/>for every policy"]
    E --> V["Recovery vs<br/>retained skill"]
```

## 9. Animation (planned)

An animated SVG of the end-to-end flow (sections 3 and 4 combined), in the style of the explainer animations and generated by a script like `docs/explainer/make_svgs.py`, is planned once the pipeline runs end to end (after M2–M3), so that it shows the real system rather than a plan.
