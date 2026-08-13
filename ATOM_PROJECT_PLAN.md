# ATOMICA — Autonomous AI Scientist for Atomic Discovery

> AI-driven scientific research system สำหรับทดลองว่า AI สามารถช่วยค้นพบ hypothesis และออกแบบ atomic experiments ใหม่ ๆ ได้หรือไม่

---

## 1. Project Overview

**ATOMICA** เป็นโปรเจกต์แนว AI + Atomic Simulation + Machine Learning ที่ไม่ได้ทำเป็น web app และไม่ได้เน้นสร้างระบบให้ความรู้ แต่เน้นการสร้าง **scientific experimentation loop** ที่สามารถรันจาก command line บนโน้ตบุ๊กได้

แนวคิดหลักคือ:

> ให้ AI ทำหน้าที่เหมือนนักวิจัยที่อ่านงานวิจัย → หา research gap → ตั้ง hypothesis → ออกแบบ experiment → เรียกใช้ atomic simulation → วิเคราะห์ผล → พยายามหักล้าง hypothesis → แล้วเลือก experiment ถัดไป

เป้าหมายไม่ใช่ให้ AI "ตอบคำถามเกี่ยวกับ Atom" แต่ให้ AI **ช่วยตัดสินใจว่าเราควรทดลองอะไรต่อ**

---

## 2. Core Research Question

คำถามหลักของโปรเจกต์:

> **Can an AI-guided scientific discovery loop discover useful or non-obvious atomic configurations more efficiently than conventional search baselines?**

คำถามรอง:

1. AI สามารถอ่าน paper และสร้าง hypothesis ที่ไม่ได้ระบุโดยตรงใน paper ได้หรือไม่?
2. AI สามารถหา unexplored research space จากงานวิจัยเดิมได้หรือไม่?
3. AI-guided search สามารถหา candidate atomic structures ได้ดีกว่า random search หรือไม่?
4. การให้ AI เรียนรู้จากผลการทดลองก่อนหน้า ช่วยให้ experiment รอบถัดไปดีขึ้นหรือไม่?
5. การมี AI critic ที่พยายามหักล้าง hypothesis ลด false discoveries ได้หรือไม่?
6. AI สามารถเลือกข้อมูลที่มีประโยชน์ต่อการ train atomic ML model ได้มีประสิทธิภาพกว่าการสุ่มหรือไม่?

---

## 3. Overall Architecture

```text
                         ┌────────────────────┐
                         │   Research Papers  │
                         │ PDF / arXiv / etc. │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Literature Agent   │
                         │ - read paper       │
                         │ - extract claims   │
                         │ - identify gaps    │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Hypothesis Agent   │
                         │ - generate ideas   │
                         │ - rank hypotheses  │
                         │ - define metrics   │
                         └─────────┬──────────┘
                                   ↓
                  ┌────────────────┴────────────────┐
                  ↓                                 ↓
        ┌──────────────────┐              ┌──────────────────┐
        │ Structure Search │              │ Dataset Search   │
        │ / Evolution      │              │ / Active Learn   │
        └────────┬─────────┘              └────────┬─────────┘
                 └──────────────┬──────────────────┘
                                ↓
                      ┌────────────────────┐
                      │ Atomic Simulator    │
                      │ ASE + MACE/CHGNet  │
                      └─────────┬──────────┘
                                ↓
                      ┌────────────────────┐
                      │ Analysis Engine    │
                      │ energy / force /   │
                      │ structure / OOD    │
                      └─────────┬──────────┘
                                ↓
                      ┌────────────────────┐
                      │ AI Critic          │
                      │ falsify / stress   │
                      │ test the claim     │
                      └─────────┬──────────┘
                                ↓
                         New Experiment
                                │
                                └──────────────↺
```

---

## 4. Fundamental Design Principle

AI และ physics/computation ต้องมีหน้าที่ต่างกันอย่างชัดเจน

### AI / Claude

รับผิดชอบ:

- Literature reasoning
- Hypothesis generation
- Research gap discovery
- Experiment planning
- Search strategy
- Result interpretation
- Criticism / falsification
- Choosing the next experiment

### Python / ML / Physics

รับผิดชอบ:

- Atomic structure manipulation
- Simulation
- Energy / force / stress calculation
- Structural optimization
- Numerical analysis
- Benchmarking
- Reproducible evaluation

### หลักการสำคัญ

> **Claude ไม่ใช่ตัวตัดสินว่าการค้นพบเป็นจริง**

ผลทางวิทยาศาสตร์ต้องมาจาก computation / simulation / measurable evidence

AI มีหน้าที่เสนอและวิจารณ์ hypothesis ส่วน evidence ต้องมาจากการทดลองที่ reproducible

---

# 5. Main Components

## 5.1 Literature Agent

Input:

```text
paper.pdf
```

Output:

```json
{
  "research_question": "...",
  "main_claims": [],
  "methods": [],
  "materials": [],
  "limitations": [],
  "unexplored_regions": [],
  "possible_extensions": []
}
```

หน้าที่:

- อ่าน paper
- สกัด research question
- ระบุ assumptions
- หาว่าทดลองอะไรแล้ว
- หาว่าอะไรยังไม่ได้ทดลอง
- สร้าง research opportunity

---

## 5.2 Hypothesis Agent

จาก literature analysis สร้าง hypothesis หลายแบบ

ตัวอย่าง:

```text
Hypothesis #1

Introducing oxygen vacancies may reduce the energy barrier
for lithium migration.

Reason:
The vacancy may create a lower-coordination migration pathway.

Expected observation:
Lower migration barrier compared with baseline.

Falsification condition:
If the relaxed structure becomes unstable or the barrier
does not decrease across independent configurations.
```

ทุก hypothesis ควรมี:

- hypothesis
- rationale
- expected observation
- measurable metric
- possible falsification condition
- proposed experiment

---

## 5.3 Structure Search Engine

รองรับหลาย strategy:

### Random Search

สร้าง baseline

```text
random structures
→ simulate
→ rank
```

### Genetic / Evolutionary Search

```text
parent
 ↓
mutation
 ↓
offspring
 ↓
simulation
 ↓
selection
 ↓
next generation
```

### AI-guided Search

AI เลือก mutation / composition / defect / geometry ที่น่าลองจากผลการทดลองก่อนหน้า

---

## 5.4 Atomic Simulation Engine

เครื่องมือหลัก:

- ASE
- MACE
- CHGNet
- PyTorch
- pymatgen

ตัวอย่าง flow:

```text
Atomic Structure
      ↓
ML potential
      ↓
Relaxation
      ↓
Energy / Forces / Stress
      ↓
Analysis
```

ภายหลังสามารถขยายไป:

- Molecular Dynamics
- Defect calculations
- Diffusion experiments
- Adsorption
- Surface structures
- Composition search
- Phase stability
- Energy landscape exploration

---

# 6. AI-guided Atomic Evolution

แนวคิด:

```text
Structure A
    ↓
AI proposes mutation
    ├── atom removal
    ├── atom addition
    ├── atom swap
    ├── element substitution
    ├── vacancy creation
    └── atomic movement
    ↓
Simulation
    ↓
Result
    ↓
AI interprets
    ↓
Next mutation
```

เปรียบเทียบ:

```text
Random Search
vs
Genetic Algorithm
vs
AI-guided Search
```

Metrics:

- Best energy found
- Number of experiments
- Search efficiency
- Structural diversity
- Compute cost
- Success rate
- Novel candidate count

---

# 7. AI Research Gap Discovery

AI ไม่ควรถามแค่ว่า:

> "paper นี้พูดอะไร?"

แต่ต้องถามว่า:

> "paper นี้ไม่ได้ทดลองอะไร?"

ตัวอย่าง:

```text
Original paper

Temperature:
300 / 500 / 700 K

Composition:
90/10
80/20

AI discovers unexplored space:

95/5 composition
Defect concentration
Local strain
Different vacancy patterns
Intermediate temperature
Surface configuration
```

จากนั้น generate experiments เพื่อสำรวจพื้นที่เหล่านั้น

---

# 8. AI Rediscovery / Paper Extension

เลือก paper จริงหนึ่ง paper

แล้วทำ experiment:

```text
Original Paper
      ↓
AI reads paper
      ↓
Reconstruct baseline
      ↓
Identify limitation
      ↓
Generate new hypothesis
      ↓
Run new experiment
      ↓
Compare with original
```

เป้าหมาย:

> ดูว่า AI สามารถสร้าง extension ที่มีเหตุผลจากงานเดิมได้หรือไม่

---

# 9. Atomic Anomaly Discovery

ใช้ machine learning หา structures ที่ผิดปกติหรือแตกต่างจาก population

ตัวอย่าง pipeline:

```text
Atomic Structures
       ↓
Structure Representation
       ↓
Embedding
       ↓
Clustering
       ↓
Anomaly Detection
       ↓
AI Interpretation
       ↓
Hypothesis
       ↓
New Experiment
```

AI ต้องตอบ:

- Structure นี้ต่างจากกลุ่มอื่นอย่างไร?
- จุดใดของ atomic arrangement น่าสนใจ?
- ความผิดปกตินั้นมี physical explanation หรือไม่?
- ควรทดลองอะไรต่อ?

---

# 10. AI Dataset Selection / Active Learning

คำถาม:

> เราจำเป็นต้องใช้ training data จำนวนมากหรือไม่ หรือ AI สามารถเลือก configuration ที่ "คุ้มค่า" ต่อการ train model ได้?

เปรียบเทียบ:

```text
Dataset A = Random selection
Dataset B = Diversity sampling
Dataset C = AI-selected structures
```

แล้ว train model กับจำนวนข้อมูลต่างกัน:

```text
100
500
1000
5000
```

วัด:

- Energy MAE
- Force MAE
- OOD performance
- Generalization
- Data efficiency

Research question:

> Can AI-selected atomic configurations produce better ML models under the same compute/data budget?

---

# 11. AI Scientist vs AI Critic

ใช้ role หลักสองแบบ:

### Scientist

สร้าง hypothesis และ experiment

### Critic

พยายามหาจุดอ่อนของ hypothesis

ตัวอย่าง:

```text
Scientist:
"Adding vacancies lowers migration barrier."

Experiment:
20 structures

Result:
3 structures improved

Critic:
The result may be confounded by composition.
The comparison set is insufficient.

New experiment:
Control composition and vary vacancy only.
```

จุดสำคัญ:

> AI ต้องมี incentive ให้ "หักล้าง" ตัวเอง ไม่ใช่พยายามทำให้ hypothesis ดูถูกต้องเสมอ

---

# 12. Persistent Research Memory

ระบบต้องเก็บประวัติการทดลองทุกครั้ง

ตัวอย่าง:

```text
memory/
├── hypotheses/
├── experiments/
├── failures/
├── discoveries/
├── critiques/
└── summaries/
```

ตัวอย่าง experiment record:

```json
{
  "experiment_id": "EXP_0037",
  "hypothesis_id": "H_0012",
  "structures_tested": 50,
  "best_candidate": "structure_183.xyz",
  "metrics": {
    "energy": -123.45
  },
  "result": "partially_supported",
  "critic_summary": "...",
  "next_action": "..."
}
```

Memory นี้ทำให้ AI รอบถัดไปไม่ต้องเริ่มคิดจากศูนย์

---

# 13. Autonomous Discovery Loop

เวอร์ชันสุดท้าย:

```text
Paper
 ↓
Literature Analysis
 ↓
Research Gap
 ↓
Hypothesis Generation
 ↓
Experiment Planning
 ↓
Structure Generation
 ↓
Simulation
 ↓
Analysis
 ↓
Critique
 ↓
Memory Update
 ↓
Next Experiment
 ↓
...
```

เป้าหมายคือให้ระบบสามารถทำหลายรอบโดยมนุษย์ทำหน้าที่เป็น supervisor

---

# 14. Baselines

เพื่อให้ project มีความเป็น research ต้องมี baseline

อย่างน้อย:

```text
Baseline 1:
Random Search

Baseline 2:
Genetic Algorithm

Baseline 3:
AI-guided Search
```

ถ้าเป็น active learning:

```text
Random Data Selection
vs
Diversity Sampling
vs
AI Selection
```

ถ้าเป็น hypothesis generation:

```text
No-memory LLM
vs
Memory-enabled AI
```

ถ้าเป็น validation:

```text
Scientist only
vs
Scientist + Critic
```

---

# 15. Evaluation Metrics

ไม่ควรใช้แค่ "ผลออกมาดูน่าสนใจ"

ต้องมี metrics ที่วัดได้

## Search Metrics

- Best energy
- Energy improvement
- Number of experiments
- Compute budget
- Improvement per experiment

## Structure Metrics

- Structural diversity
- Novelty
- Similarity to known structures
- Stability

## AI Metrics

- Hypothesis acceptance rate
- Useful hypothesis rate
- False discovery rate
- Experiment efficiency
- Critic rejection rate

## ML Metrics

- Energy MAE
- Force MAE
- Stress MAE
- OOD performance
- Data efficiency

---

# 16. Project Repository

แนะนำโครงสร้าง:

```text
atomica/
│
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
│
├── papers/
│
├── experiments/
│
├── structures/
│
├── simulations/
│
├── results/
│
├── memory/
│   ├── hypotheses/
│   ├── experiments/
│   ├── critiques/
│   └── discoveries/
│
├── agents/
│   ├── literature_agent.py
│   ├── hypothesis_agent.py
│   ├── experiment_agent.py
│   ├── analysis_agent.py
│   └── critic_agent.py
│
├── simulation/
│   ├── ase_utils.py
│   ├── mace_runner.py
│   └── chgnet_runner.py
│
├── search/
│   ├── random_search.py
│   ├── genetic_search.py
│   └── ai_guided_search.py
│
├── analysis/
│   ├── structures.py
│   ├── anomaly.py
│   └── metrics.py
│
├── benchmarks/
│
└── run.py
```

---

# 17. CLI Concept

ไม่ทำ web app

ใช้ command line:

```bash
python run.py --paper papers/paper_001.pdf
```

หรือ:

```bash
python run.py \
  --goal "find low-energy atomic configurations" \
  --budget 100
```

ดูสถานะ:

```bash
python run.py --status
```

รัน experiment เฉพาะ:

```bash
python run.py --experiment ai_search
```

ดู research history:

```bash
python run.py --history
```

---

# 18. Example Run

```text
ATOMICA
=======

Paper:
paper_001.pdf

Goal:
Find non-obvious low-energy configurations.

--------------------------------------------------

LITERATURE AGENT

Main claim:
...

Unexplored region:
...

--------------------------------------------------

HYPOTHESIS AGENT

H1:
Introducing vacancy X may lower local energy.

H2:
Substitution A → B may stabilize coordination.

H3:
Local strain may create a new low-energy basin.

--------------------------------------------------

EXPERIMENT AGENT

Budget:
100 simulations

Strategy:
AI-guided search

--------------------------------------------------

RUNNING...

EXP 001 / 100
EXP 002 / 100
EXP 003 / 100
...

--------------------------------------------------

ANALYSIS

Interesting candidates:
7

Best candidate:
structure_183.xyz

Energy improvement:
...

--------------------------------------------------

AI CRITIC

H1:
Partially supported

Main concern:
Insufficient control experiment.

Recommendation:
Run controlled vacancy experiment.

--------------------------------------------------

NEXT EXPERIMENT

EXP 101+
Controlled vacancy sweep

--------------------------------------------------
```

---

# 19. Development Roadmap

## Phase 0 — Environment

Goal:

ทำให้ atomic simulation รันได้

Tasks:

- Python environment
- ASE
- MACE หรือ CHGNet
- PyTorch
- pymatgen
- basic atomic structure I/O

Success condition:

```text
structure
→ relax
→ energy
→ save result
```

---

## Phase 1 — Atomic Laboratory

สร้าง simulation engine ที่ reproducible

Tasks:

- structure generation
- relaxation
- energy calculation
- batch experiments
- result logging

Success condition:

```text
100 structures
→ simulation
→ ranked results
```

---

## Phase 2 — AI Scientist

เพิ่ม:

- literature analysis
- hypothesis generation
- experiment planning

Success condition:

```text
paper
→ hypothesis
→ experiment.json
```

---

## Phase 3 — Search

เพิ่ม:

- random search
- genetic search
- AI-guided search

Success condition:

สามารถ benchmark 3 วิธีได้

---

## Phase 4 — Research Memory

เพิ่ม:

- experiment history
- failure memory
- hypothesis memory
- result summaries

Success condition:

experiment รอบใหม่สามารถใช้ข้อมูลจากรอบก่อนหน้าได้

---

## Phase 5 — AI Critic

เพิ่ม scientific falsification loop

Success condition:

ระบบสามารถ detect weak hypotheses / insufficient experiments ได้

---

## Phase 6 — Autonomous Loop

รวมทุกอย่าง:

```text
Research
→ Hypothesis
→ Experiment
→ Simulation
→ Analysis
→ Critique
→ Memory
→ Next Experiment
```

Success condition:

ระบบสามารถทำ iterative discovery ได้หลายรอบโดย intervention ต่ำ

---

# 20. First Research Experiment

ไม่ควรเริ่มจากโจทย์ที่ใหญ่เกินไป

เลือก problem เล็กที่ simulation เร็ว

ตัวอย่าง:

- vacancy stability
- simple substitution
- low-energy configuration search
- atomic rearrangement
- small diffusion problem
- small surface adsorption problem

แล้วทำ benchmark:

```text
100 experiments

Random Search
vs
Genetic Search
vs
AI-guided Search
```

คำถาม:

> ภายใต้ compute budget เท่ากัน วิธีไหนหา candidate ที่ดีที่สุดได้เร็วที่สุด?

นี่ควรเป็น experiment แรก เพราะวัดผลได้ชัด

---

# 21. Second Research Experiment

เอา paper จริง 1 paper

ทำ:

```text
Paper
↓
AI extracts baseline
↓
AI finds unexplored region
↓
Generate candidate experiments
↓
Run simulation
↓
Compare with paper
```

คำถาม:

> AI สามารถเสนอ extension ที่น่าสนใจจากงานวิจัยเดิมได้หรือไม่?

---

# 22. Third Research Experiment

เพิ่ม critic:

```text
AI Scientist
vs
AI Scientist + Critic
```

เปรียบเทียบ:

- false discoveries
- unnecessary experiments
- hypothesis quality
- compute efficiency

คำถาม:

> การมี AI critic ช่วยลด false scientific conclusions ได้หรือไม่?

---

# 23. Final Vision

ATOMICA ไม่ควรเป็นแค่:

```text
LLM + Atomic Simulation
```

แต่ควรเป็น:

```text
LLM
+
Scientific Literature
+
Atomic Simulation
+
Machine Learning
+
Search
+
Memory
+
Criticism
+
Experimentation
```

จนกลายเป็น:

> **An AI system that iteratively reads scientific knowledge, proposes hypotheses, performs computational atomic experiments, evaluates evidence, and decides what to investigate next.**

---

# 24. Guiding Principle

โปรเจกต์นี้ควรยึดหลัก:

> **AI should not only answer scientific questions.  
> AI should help decide which scientific question to test next.**

และ:

> **The goal is not to make AI sound like a scientist.  
> The goal is to make AI participate in a reproducible scientific loop.**

---

# 25. Potential Final Project Statement

> **ATOMICA is an autonomous AI-driven research framework for atomic discovery. It combines large language models, scientific literature, atomistic machine-learning potentials, structural search, persistent experiment memory, and AI-based scientific criticism to iteratively generate and test hypotheses about atomic systems.**

---

# 26. Potential Thesis / Paper Titles

1. **ATOMICA: An Autonomous AI Scientist for Atomic Discovery**
2. **LLM-Guided Search for Novel Atomic Configurations**
3. **From Scientific Papers to Atomic Experiments: An AI-Driven Discovery Loop**
4. **Can Language Models Guide Atomic Structure Discovery More Efficiently Than Random Search?**
5. **AI-Guided Scientific Exploration of Atomic Configuration Space**
6. **An LLM-Based Hypothesis–Experiment–Critique Loop for Atomistic Research**

---

# 27. Recommended First Milestone

อย่าเริ่มด้วย autonomous system เต็มรูปแบบ

Milestone แรกควรเป็น:

```text
Input:
1 paper
+
1 research goal

Output:
10 AI-generated hypotheses
+
100 atomic experiments
+
ranked candidate structures
+
AI critique
+
next experiment recommendation
```

ถ้าทำ milestone นี้ได้สำเร็จ แปลว่า architecture หลักของ ATOMICA เริ่มใช้งานได้จริงแล้ว
