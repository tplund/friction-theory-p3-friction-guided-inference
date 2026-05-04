# Pipeline Audit — Hårde regler

**Formål**: Dette dokument definerer hvad pipeline audit'en SKAL gøre
og hvad den IKKE MÅ antage. Alt kode der implementerer audit'en skal
overholde disse regler. Reglerne er lært fra fejl i sessionen
2026-04-12/13.

---

## Regel 1: Ingen antagelser fra andre modeller eller benchmarks

- Tærskler, signaler, strategier, og stop-kriterier fra én model×benchmark
  GÆLDER IKKE for en anden.
- Audit'en SKAL beregne ALT fra den aktuelle calibrations-data.
- "Det virkede på cogito" er IKKE et argument for at gøre det på 405B.
- Selv "det virkede på cogito×GPQA" er IKKE et argument for cogito×SimpleQA.

## Regel 2: Auto-sidecar — beregn tærskler fra DENNE data

- Audit'en SKAL beregne p33/p50/p67 for mean_cr og std_cr fra
  calibrations-data.
- Eksisterende sidecars er KUN referencer til at detektere drift.
- Hvis drift > 0.1: ADVAR og brug de NYE tærskler.
- Sidecar-refresh SKAL tilbydes som output (--update-sidecar).

## Regel 3: Cell-dimensionalitet er empirisk — 2D er IKKE givet

- Audit'en SKAL teste 1D, 2D, og 3D grids.
- Akserne er IKKE forudbestemt (ikke nødvendigvis magnitude×location).
- ALLE features er kandidat-akser.
- Vinderen vælges af: "hvilken grid giver højest per-cell strategi-routing
  accuracy på et held-out test-split?"
- Resultatet kan være 1D (én akse er nok) eller 3D (tre akser giver
  bedre separation). Data bestemmer.

## Regel 4: Profil-kategorier er IKKE forudbestemte

- SPIKY/EVEN/SPREAD/LOW er én mulig kategorisering.
- Audit'en SKAL teste om tercile-splits på vilkårlige features giver
  bedre strategi-routing end de foruddefinerede kategorier.
- Hvis tercile-split på "trajectory" giver bedre accuracy end
  SPIKY/SPREAD-split, skal audit'en anbefale trajectory-split.

## Regel 5: Alle strategier testes — inkl. pre-mortem og reexamine

- Calibration SKAL køre ALLE tilgængelige strategier:
  vanilla, step_by_step, challenge, narrow, verify,
  pre_mortem, pre_mortem_rescue, reexamine.
- pre_mortem, pre_mortem_rescue og reexamine er FOLLOW-UP strategier:
  de SKAL se vanilla-svaret som kontekst (multi-turn prompt).
- Standalone strategier (sbs, challenge, narrow, verify) ser IKKE
  vanilla-svaret.
- Audit'en SKAL rapportere per-strategi: any_correct, unique_correct,
  rescues, regressions, net (constructive/destructive/neutral).

## Regel 6: Konstruktiv vs destruktiv er PER MODEL × PER BENCHMARK

- Challenge er konstruktiv på cogito×GPQA men destruktiv på 405B×GPQA
  og qwen3_235b×SimpleQA.
- step_by_step er destruktiv på cogito×GPQA men konstruktiv på
  qwen3_235b×SimpleQA.
- Audit'en SKAL klassificere HVER strategi som
  constructive/destructive/neutral for DENNE model×benchmark.
- Adaptive pipeline SKAL ekskludere destruktive strategier.
- Default queue SKAL kun indeholde constructive strategier.

## Regel 7: Stop-kriterie er lært, ikke hardcoded

- MAX_DEPTH er kun en stopgap (sæt til 20).
- Calibration SKAL køre nok runder til at oracle plateauer
  (3+ runder uden nye rescues).
- Brug --cal-cycles til at cykle strategier flere gange.
- Audit'en SKAL teste mindst: fixed_all_MCW, agreement_2,
  mode_of_unique, decay_MCW, best_cr_round, last_round.
- Vinderen bruges i adaptive pipeline.
- Hvis oracle stadig stiger ved sidste runde: dette er en
  BLOCKER — ikke en advarsel. Du SKAL:
  1. STOPPE. Kør IKKE adaptive på ufuldstændig kalibrering.
  2. Køre dybere kalibrering (--cal-cycles 3 eller højere,
     op til 20 runder totalt) indtil oracle plateauer.
  3. Køre audit IGEN på den dybere kalibrering.
  4. FØRST NÅR oracle er plateauet: gå videre til adaptive.
- "Oracle still climbing" + "kør adaptive alligevel" er en
  FEJL vi har lavet gentagne gange. Det giver forkerte
  strategi-vurderinger og underestimerer headroom.
- Plateau-kriteriet SKAL være statistisk: kræv mindst 3
  CONSECUTIVE runder med 0 nye rescues. Én runde med 0
  efter en runde med +5 er IKKE et plateau — det kan være noise.
  Først når 3 runder i træk giver 0 nye: plateau bekræftet.

### 7b: Dybde-kalibrering: SAMME strategi gentaget (MANDATORY)
- Kalibrering SKAL teste DYBDE (gentag bedste strategi N gange),
  IKKE kun bredde (N forskellige strategier).
- Bredde finder HVILKE strategier der virker.
  Dybde finder HVOR MANGE gange man skal køre dem.
- Kør bedste strategi (f.eks. SBS) gentaget op til 20 gange
  på et subset (50-100q). Oracle SKAL plateau inden adaptive.
- Hypotese: alle retrievable spørgsmål (modellen VED svaret
  men committer ikke) kan recoveres ved nok gentagelser.
  Test denne hypotese eksplicit — rapporter oracle per runde.
- DYBDE-DATA SKAL SKRIVES TIL SEPARAT FIL — brug sidecar-id
  med suffix `_depth` (f.eks. `--sidecar-id qwen3_235b_depth`).
  Bredde og dybde deler idx-range, og resume-logik springer
  over spørgsmål der allerede er i output. Hvis begge skriver
  til samme fil, blandes formaterne (8 runder vs 21 runder)
  og resume skipper dybde-spørgsmålene.

### 7c: Pre-mortem som destabilizer for confident-wrong
- Retrievable = modellen ved svaret men committer ikke → SBS gentaget.
- Confident-wrong = modellen er sikker men FORKERT → SBS hjælper IKKE.
- For confident-wrong: kør pre-mortem som destabilizer FØRST,
  derefter SBS igen. Pre-mortem skaber positiv friktion der
  destabiliserer den forkerte overbevisning.
- Kalibrering SKAL teste denne sekvens:
  1. Kør SBS N gange → find de spørgsmål der forbliver forkerte
  2. Kør pre-mortem på de forkerte
  3. Kør SBS igen efter pre-mortem
  4. Rapporter: hvor mange recoverede pre-mortem IKKE alene
     løste, men som SBS-efter-pre-mortem REDDEDE?
- Dette er den fulde pipeline: SBS×N → pre-mortem → SBS×M.

## Regel 8: Commit-mekanisme er ikke nødvendigvis MCW

- MCW (majority CR-weighted vote) er ÉN mulig commit-mekanisme.
- mode_of_unique (simpel flertalsafstemning) slog MCW på cogito.
- agreement_2 (stop tidligt, MCW) slog begge på qwen3_235b.
- Audit'en SKAL teste alle og anbefale den bedste.
- Adaptive pipeline SKAL bruge den anbefalede commit-mekanisme.

## Regel 9: Oscillation SKAL detekteres og håndteres

- Audit'en SKAL rapportere oscillation-rate (% af spørgsmål der
  flipper 2+ gange mellem rigtigt og forkert).
- Hvis oscillation > 20%: ADVAR.
- Per-strategi regression-rate SKAL rapporteres: hvilke strategier
  FORÅRSAGER at et rigtigt svar bliver forkert?
- Destruktive strategier der forårsager regressions SKAL ekskluderes.

## Regel 10: Hysterese-test er obligatorisk per model×benchmark

- Kør 20 spørgsmål med strategier i FORWARD og REVERSED rækkefølge.
- Sammenlign per-strategi accuracy og oracle.
- Hvis oracle-delta > 5pp: rækkefølgen BETYDER NOGET og strategier
  SKAL randomiseres i calibration.
- En negativ hysterese-test på én model overføres IKKE til en anden.

## Regel 11: Disagreement-signal SKAL testes

- Strategi-uenighed (antal unikke predictions på tværs af runder)
  er et cross-model signal for retrievable headroom.
- Retrievable spørgsmål har FLERE unikke predictions end epistemic.
- Audit'en SKAL rapportere disagreement-metrics for retrievable vs
  epistemic grupper.
- Hvis signalet er stærkt (>10% separation): anbefal
  disagreement-gated routing.

## Regel 12: Retrievable vs epistemic split

- Retrievable = vanilla forkert, oracle rigtigt (modellen VED svaret
  men committer ikke).
- Epistemic = vanilla forkert, oracle også forkert (modellen VED
  IKKE svaret).
- Audit'en SKAL rapportere: n_retrievable, n_epistemic,
  retrievable_fraction, retrievable_headroom_pp.
- ALLE features SKAL testes for separation (retrievable vs epistemic).
- Train/test classifier SKAL valideres blindt.

## Regel 13: Grading, logprobs-format og judging SKAL verificeres i audit

### 13a: Grading-metode
- Audit'en SKAL verificere og rapportere HVILKEN grading-metode
  der er brugt i kalibreringsdata (`grading_method` felt).
- For benchmarks med officiel grading-metode:
  - SimpleQA: LLM-judge (OpenAI bruger det selv)
  - GPQA Diamond: letter match (MC)
  - MATH: LLM-judge eller exact match
  - TruthfulQA MC: letter match
- Hvis kalibreringsdata bruger en ANDEN metode end den officielle:
  ADVAR og anbefal re-grading FØR audit-analyse.
- KRITISK: Kalibreringsdata SKAL grades med SAMME metode som
  adaptive/paper-results. Strategi-klassificering (konstruktiv/
  destruktiv) er kun så god som dens labels.

### 13b: Logprobs-format
- Audit'en SKAL verificere at logprobs-data er korrekt:
  - `per_token_cr` er IKKE tom og har værdier > 1.0
  - `raw_top_logprobs` har top_logprobs count > 1 per token
  - Formatet matcher forventet (OpenAI content array vs legacy)
- Hvis CR-værdier er alle 1.000: FEJL — logprobs=1 (integer)
  blev brugt i stedet for logprobs=True (boolean).
- Hvis raw_top_logprobs er tom eller mangler: ADVAR — kan ikke
  genberegne CR eller entropy fra rå data.
- Audit'en SKAL rapportere: logprobs_format, avg_top_logprobs_count,
  pct_cr_above_1, pct_empty_logprobs.

### 13c: Judging-pipeline
- For open-ended benchmarks (SimpleQA, MATH): LLM-judge SKAL
  være en integreret del af pipeline, IKKE en efterfølgende tanke.
- Workflow: generate → judge → audit → adaptive → judge.
- Judge-modellen SKAL dokumenteres (claude-haiku, gpt-4o, etc.)
- Post-hoc re-grading er acceptabelt for eksisterende data,
  men nye kalibreringer SKAL grade inline.
- For teknisk smoke-test: string-match er OK.
  For ALLE strategi-beslutninger: officiel grading-metode.

### 13d: Judge-bias mod lange svar (KRITISK)
- LLM-judges har en systematisk bias mod step-by-step svar:
  korte svar ("Edward Teller") vurderes nemt korrekt, mens
  lange svar med ræsonnement straffes fordi svaret "drukner"
  i teksten eller ræsonnementet ser ufærdigt ud.
- Vi fandt 54/500 SBS-svar der indeholdt gold-svaret men blev
  judget INCORRECT, mod kun 3/500 for vanilla. Det er 18x bias.
- Denne bias FORVRIDER strategi-klassificering: SBS så ud til
  net +29 men var reelt net +75.
- Judge-prompten SKAL instruere judgen om at:
  (a) Søge i HELE response-teksten, ikke kun det endelige svar
  (b) Acceptere korrekt svar indlejret i ræsonnement
  (c) Ikke straffe ufærdigt ræsonnement hvis svaret er der
- VALIDERING (MANDATORY): Efter judging, kør substring-match som
  kryds-tjek. Rapporter antal tilfælde hvor judge siger INCORRECT
  men gold ER i response-teksten. Hvis >2% for nogen strategi:
  ADVAR om judge-bias og undersøg prompten.
- Bias-tjekket SKAL være PER STRATEGI, ikke aggregeret. Bias
  rammer strategier med lange svar (SBS, pre-mortem, reexamine)
  hårdere end korte (vanilla, narrow).

### 13e: max_tokens skal matche svar-format
- Hvis en strategi producerer længere svar end vanilla (SBS,
  pre-mortem, reexamine), SKAL max_tokens være høj nok til at
  svaret ikke trunkeres.
- Trunkerede svar giver falske INCORRECT fra judge OG forvrider
  CR-signalet (ufærdigt svar har anderledes CR-fordeling).
- SimpleQA vanilla: ~10 tokens. SBS: ~150 tokens. max_tokens
  SKAL være mindst 512 for open-format med SBS.
- Audit'en SKAL rapportere per-strategi: pct_at_max_tokens.
  Hvis >5% for nogen strategi: ADVAR og anbefal højere limit.

## Regel 14: Alt gemmes — rå data, streaming, resume

### 14a: Rå data per token (MANDATORY)
- HVERT API-kald SKAL gemme ALLE disse felter:
  - `per_token_cr`: rå CR-liste per token (f.eks. [1, 2, 1, 3, 1, ...])
  - `per_token_entropy`: Shannon entropy per token
  - `raw_top_logprobs`: fuld top_logprobs-liste (min 20 tokens)
  - `mean_cr`, `std_cr`, `n_tokens`: aggregater
  - `response`: fuld model-svar (truncate til 500-1000 chars)
  - `predicted`: ekstraheret svar
  - `correct`: om svaret er korrekt (inkl. grading-metode)
  - `strategy`: strategi-navn
- De 7 signaler (cr, std, spikes, first5, last5, skew, response_len)
  SKAL kunne genberegnes fra rå data.
- ALDRIG gem kun aggregater — gem altid rå lister.

### 14b: Streaming og resume (MANDATORY)
- ALLE scripts SKAL streame resultater line-by-line til JSONL.
  Skriv én linje per spørgsmål, flush efter hver linje.
- ALLE scripts SKAL supportere resume: check done IDs ved opstart,
  skip dem. Muliggør genoptagelse efter crash/timeout.
- ALLE scripts SKAL logge progress til console:
  `[n/total] q{id} van=Y/. adp=Y/. ora=Y/. (Xs)`

### 14c: Hvad der gemmes hvor
- Calibrations-data: JSONL med alle felter fra 14a per runde.
- Audit-rapport: JSON med alle findings, warnings, recommended_config.
- Sidecar: JSON med tærskler beregnet fra denne data.
- Adaptive resultater: JSONL med rounds, commit_method, stop_reason.
- INTET antages at kunne genberegnes — gem alt.
- Storage er billigt, re-running er dyrt.

## Regel 15: Audit-rapport er input til adaptive

- Adaptive pipeline SKAL acceptere --audit-report parameter.
- Adaptive SKAL bruge audit-rapportens:
  - constructive_strategies (for default queue)
  - destructive_strategies (for ekskludering)
  - best_stop (for stop-kriterie)
  - best_grid (for cell-dimensionalitet)
  - sidecar (for tærskel-kalibrering)
- Adaptive UDEN audit-rapport SKAL advare tydeligt.

## Regel 16: Platform-quirks dokumenteres

- Together.ai og Fireworks har forskellige logprobs-formater
  (legacy vs OpenAI content array).
- Together.ai kan have server-side caching.
- Model-drift kan ske mellem kørsler (cogito vanilla 44% → 66%).
- Platform og model_id SKAL gemmes i HVER resultat-fil.
- Calibration og adaptive SKAL køre i SAMME session for at undgå
  drift mellem runs.

## Regel 17: Model-tilgængelighed verificeres FØR kalibrering

- Modeller kan FORSVINDE fra serverless (Qwen3-235B-A22B fjernet
  2026-04-13 uden varsel).
- FØR kalibrering: verificer at model_id er tilgængelig med et
  simpelt API-kald (1 spørgsmål, tjek status 200).
- Hvis modellen kræver dedicated endpoint: STOP og spørg om
  godkendelse inden oprettelse.
- Kalibrering på en model der ikke er tilgængelig til adaptive
  er SPILD.
- Dokumenter model_id versioning (fp8, -tput, -2507 etc.) —
  kvantisering og version KAN ændre CR-fordeling.

## Regel 18: Grading-konsistens mellem kalibrering og adaptive

- Grading-metoden SKAL være IDENTISK i kalibrering og adaptive.
- Hvis adaptive grades med LLM-judge, SKAL kalibrering OGSÅ
  grades med LLM-judge (ellers er strategi-klassificering forkert).
- Grading-metoden SKAL dokumenteres i resultat-filen som et felt
  (`grading_method`: "string_match" | "llm_judge_claude" | etc.)
- For benchmarks med officiel grading-metode (SimpleQA: LLM-judge,
  GPQA: letter match, MATH: LLM-judge): brug SAMME metode som
  officiel for at være sammenlignelig.
- IDEELT: LLM-judge grading SKAL være integreret i kalibrerings-
  scriptet (ikke en separat post-hoc step). Post-hoc re-grading
  er acceptabelt for eksisterende data, men nye kalibreringer
  SKAL grade inline.
- `grading_method` feltet SKAL gemmes i HVER record, IKKE kun
  som metadata. Muliggør blandet grading i samme fil.

## Regel 19: Smoke → pilot → fuld kørsel (trinvis eskalering)

- ALDRIG kør direkte fra kalibrering til fuld kørsel (N>1000q).
- Trinvis eskalering:
  1. **Smoke** (3-5q): teknisk — virker API, logprobs, grading?
  2. **Pilot** (100q): faglig — er der noget at komme efter?
     Tjek accuracy, strategi-effekt, fejlmønstre.
     PAUSE og review med menneske.
  3. **Fuld kørsel**: kun efter pilot er godkendt.
- Pilot-resultater SKAL inspiceres manuelt: er lift reelt?
  Er fejlmønstrene som forventet? Er der systematiske problemer?
- Brug `--limit 100` til pilot.

## Regel 20: Zero-effect rule

- Hvis NOGET eksperiment viser PRÆCIS 0 effekt, 0 korrelation,
  alle-same scores, eller perfekt uniform fordeling:
  ANTAG EN BUG indtil bevist ellers.
- Undersøg FØR rapportering. Almindelige årsager:
  - logprobs=1 i stedet for logprobs=True
  - Forkert felt-navn (vanilla_cr vs competing_routes_score)
  - API returnerer cached/deterministiske svar
  - Score-parsing fejler lydløst (returnerer 0 ved parse error)
  - Kalibrering med anderledes max_tokens end eksperiment

## Regel 22: ALDRIG overskrive eksisterende data

- `--fresh` SLETTER eksisterende data. Brug det KUN på test/smoke.
- Nye kørsler SKAL skrive til NYE filer med beskrivende navne:
  - Dybdekalibrering: `cal_depth_{benchmark}_{sidecar}.jsonl`
  - Re-graded: `cal_{benchmark}_{sidecar}_judged_v2.jsonl`
  - Piloter: `pilot_{benchmark}_{sidecar}_{dato}.jsonl`
- Kalibreringsdata der har taget timer at generere SKAL ALTID
  committes til git FØR nye eksperimenter startes.
- Hvis du er i tvivl: ALDRIG `--fresh` på en fil der ikke er i git.

## Regel 21: Format-match mellem profiling og evaluering

- CR profiling SKAL bruge SAME format som evaluering.
- MC benchmarks (GPQA, TQA): profil med MC prompt + logprobs.
- Open benchmarks (SimpleQA, MATH): profil med open format.
- ALDRIG profil med open format og evaluer med MC — CR-fordelinger
  er fundamentalt forskellige.

## Regel 23: Pre-flight tjek INDEN ethvert eksperiment (BLOCKER)

Ethvert eksperiment der koster penge (API-kald) SKAL bestå ALLE
pre-flight checks FØR det starter. Ingen undtagelser.

### 23a: Kode SKAL være committed til git
- `git status` SKAL vise clean working tree for ALLE scripts
  der indgår i eksperimentet.
- Hvis et script er ændret men ikke committed: STOP.
  Commit først, kør derefter.
- Begrundelse: hvis noget går galt, kan vi inspicere præcis
  den kode der kørte. Ellers debugger vi kode der ikke
  matcher det der faktisk kørte.

### 23b: Automatiserede integritets-tests (MANDATORY)
Kør `python scripts/sanity_check.py` med MINIMUM disse checks:

1. **Oracle-cheat test**: Verificer at commit-mekanismen
   ALDRIG ser ground truth. Kør 5 spørgsmål med falske
   gold-labels. Hvis accuracy > 0%: koden snyder.
   Vi mistede 2 dages arbejde + penge på en pipeline der
   kiggede på ground truth og commitede det rigtige svar.

2. **Output-format test**: Kør 3 spørgsmål, verificer at
   output-JSONL har ALLE påkrævede felter:
   per_token_cr, per_token_entropy, raw_top_logprobs,
   response, predicted, correct, strategy, grading_method.

3. **Logprobs-kvalitet**: Verificer at per_token_cr har
   værdier > 1.0 (ikke alle 1.000) og at raw_top_logprobs
   har count > 1 per token.

4. **Commit-mekanisme purity**: Verificer at committed_pred
   KUN afhænger af model-output og CR, ALDRIG af gold/answer.
   Grep koden for steder hvor `gold`, `answer`, `correct`
   bruges i commit-logikken.

5. **Fil-overskrivelses-tjek**: Verificer at output-filen
   ikke allerede eksisterer med data vi vil miste.
   Hvis den gør: STOP og brug et nyt filnavn.

### 23c: Menneske-review af eksperiment-plan
- FØR ethvert eksperiment: skriv en 3-linje plan:
  1. Hvad kører vi (model, benchmark, n spørgsmål, strategi)
  2. Hvad koster det (estimeret)
  3. Hvad forventer vi at finde (baseline og forventet lift)
- Menneske godkender planen inden start.

### 23d: Eksperiment-template (alle scripts SKAL følge)
- Alle scripts der laver API-kald SKAL:
  - Importere fra `friktionsllm.friction.cr_utils` (ikke egne CR-beregninger)
  - Bruge `logprobs=True` + `top_logprobs=5` (ALDRIG logprobs=1)
  - Streame resultater JSONL line-by-line med flush
  - Supportere resume (check done IDs)
  - Logge progress: `[n/total] id: result (Xs)`
  - Gemme ALLE felter fra R14a
  - Gemme grading_method per record
  - ALDRIG referere ground truth i commit-logik

### 23e: Oracle-cheat check SKAL være semantisk, ikke kun syntaktisk
- Pre-flight SKAL tjekke ALLE variabelnavne der bruges i
  commit/routing logik mod en bred liste: gold, answer, correct,
  ground_truth, reference, gold_letter, is_correct, final_correct,
  answer_key, label, target, expected.
- IKKE kun i funktioner der hedder "commit" — også i while-loops,
  if-statements, og lambda-funktioner der styrer routing.
- Pre-flight SKAL også tjekke om committed_correct beregnes fra
  ground truth (trace dataflow, ikke kun variabelnavne).
- Begrundelse: Oracle-cheat buggen brugte `final_correct` som
  SÅ ud som output men var beregnet fra gold. Ren navne-check
  fanger det ikke.

### 23f: --fresh SKAL have sikkerhedsnet i koden
- `--fresh` SKAL ALTID tjekke: eksisterer output-filen OG er den
  IKKE i git? Hvis ja: AFVIS med fejlbesked.
- For at overskrive data der ikke er i git: kræv eksplicit
  `--fresh --force-overwrite` (dobbelt flag).
- Alternativt: `--fresh` laver backup (rename til .bak) i stedet
  for at slette.
- Begrundelse: vi mistede 198q kalibrering med `--fresh`.

### 23g: Adaptive SKAL KRÆVE audit-rapport (ikke optional)
- `--audit-report` SKAL være REQUIRED i adaptive mode, ikke optional.
- Adaptive UDEN audit-rapport SKAL FEJLE med klar besked:
  "Cannot run adaptive without audit report. Run pipeline_audit.py first."
- Audit-rapporten SKAL have status=COMPLETE. Hvis INCOMPLETE: AFVIS.
- ALDRIG manuelt override audit-status fra INCOMPLETE til COMPLETE.
  Hvis audit siger BLOCKER, FIX den underliggende årsag (kør dybde,
  re-grade, etc.) og kør audit IGEN. Manuel override omgår kvalitets-
  systemet og vi har gjort det — det er forkert.
- Begrundelse: Workflow-rækkefølgen håndhæves ingen steder i kode.
  Man kan køre adaptive uden audit og få forkerte resultater.

### 23h: Grading-konsistens håndhæves i audit
- Audit SKAL tjekke om kalibrerings-data og audit-rapport bruger
  SAMME grading_method.
- Audit SKAL gemme grading_method i recommended_config.
- Adaptive SKAL tjekke at grading_method i audit-rapport matcher
  den der bruges i adaptive.
- For open benchmarks: audit SKAL advare hvis grading_method
  IKKE er LLM-judge.

### 23i: max_tokens SKAL matche mellem kalibrering og adaptive
- Audit SKAL rapportere max_tokens brugt i kalibrering (infereret
  fra max(n_tokens) per strategi).
- Adaptive SKAL bruge MINDST samme max_tokens som kalibrering.
- Hvis adaptive bruger HØJERE max_tokens end kalibrering:
  ADVAR — CR-profiler kan ændre sig med længere svar.

### 23j: Pre-flight SKAL tjekke ALLE scripts i pipelinen
- Pre-flight SKAL tjekke ALLE scripts der indgår i eksperimentet,
  ikke kun ét ad gangen.
- Default: tjek run_iterative_adaptive_any_benchmark.py +
  pipeline_audit.py + judge_simpleqa_cal.py (hvis open benchmark).
- Alle SKAL være committed, alle SKAL bestå checks.

---

## Workflow (den rigtige rækkefølge)

```
 0. Pre-flight check (python scripts/preflight_check.py) → SKAL PASSE
 1. Verificer model-tilgængelighed (1 API-kald)
 2. Hysterese-test (20q, forward + reversed) → dokumentér
 3. KALIBRERING — to faser i ÉN kørsel:
    a) BREDDE: alle 8 strategier × 1 cycle → finder hvad der virker
    b) DYBDE: bedste strategi gentaget 20× → finder oracle-plateau
    Brug: --cal-cycles 3 for bredde, --depth-strategy X for dybde
    Kør begge og KOMBINÉR i samme output-fil.
 3b. LLM-judge grading af cal-data (for open benchmarks)
 3c. Pre-mortem destabilizer-test på spørgsmål der forbliver
     forkerte efter dybde: pre-mortem → bedste strategi igen
 4. Commit kalibrerings-data til git
 5. Pipeline audit → JSON rapport
    Audit TJEKKER automatisk:
    - Er dybde-data inkluderet? (>=3 gentagelser af bedste strat)
    - Er oracle plateauet? (3+ runder uden ny rescue)
    - Er grading korrekt? (judge-bias check per strategi)
    Hvis NOGEN af disse fejler: rapport siger INCOMPLETE
 6. Review audit-rapport (menneske!) → godkend
    BLOCKER: Hvis status=INCOMPLETE → gå IKKE videre.
    Fiks det der mangler og kør audit igen.
 7. Pre-flight check IGEN
 8. Smoke test (3-5q med audit-rapport) → teknisk OK?
 9. Pilot (100q med audit-rapport) → faglig review
10. Review pilot (menneske!)
11. Fuld adaptive kørsel (med --audit-report)
12. LLM-judge grading af resultater (for open benchmarks)
13. Commit ALLE resultater til git
```

Step 6 og 10 er GATES: et menneske SKAL reviewe.
Audit status=INCOMPLETE er en BLOCKER — kør ALDRIG adaptive
på ufuldstændig kalibrering.
Pre-flight check (step 0, 7) er automatiserede BLOCKERS.
Git-commits (step 4, 13) er MANDATORY — vi mistede 198q data.

---

## Fejl vi har lavet (og som disse regler forhindrer)

| Fejl | Regel |
|---|---|
| Kopierede qwen25 MATH config til 405B GPQA | R1 |
| Brugte gammel sidecar på driftet model | R2 |
| Antog 2D celler var nok uden at teste | R3, R4 |
| Kørte aldrig pre-mortem i calibration | R5 |
| Antog challenge var konstruktiv på alle modeller | R6 |
| MAX_DEPTH=4-5 skjulte epistemic ceiling | R7 |
| MCW var flad pga oscillation | R8, R9 |
| Antog hysterese-test fra cogito gjaldt qwen3 | R10 |
| Reexamine som standalone (ikke multi-turn) | R5 |
| SimpleQA string-match grading for paper-claims | R13 |
| Adaptive ignorerede audit-resultater | R15 |
| Model fjernet fra serverless midt i projekt | R17 |
| Cal brugte string-match, adaptive ville bruge judge | R18 |
| Ville køre 4326q direkte uden pilot | R19 |
| Ordering test gemte kun mean_cr, tabte rå data | R14a |
| Judge-bias: 54 SBS-svar med rigtigt svar judget INCORRECT | R13d |
| SBS trunkeret ved 256 tokens, svar afskåret | R13e |
| Judge-bias forvred strategi-net fra +75 til +29 | R13d |
| Ignorerede "oracle still climbing" og kørte adaptive alligevel | R7 |
| --fresh overskrev 198q kalibrering der aldrig var committed | R14, R22 |
