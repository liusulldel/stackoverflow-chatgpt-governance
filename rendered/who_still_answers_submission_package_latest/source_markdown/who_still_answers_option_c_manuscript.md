# Private AI, Public Answer Work, and Certification at Stack Overflow

## Abstract

Private AI creates a new outside option for technical problem solving. Its effects on public knowledge platforms may therefore appear less in activity counts than in which questions still reach the public archive, who performs visible answer work inside that residual queue, and how that work becomes certified public resolution.

We analyze the official Stack Overflow `2025Q4` dump: `2,035,885` single-focal-tag questions and `2,391,883` matched answers across `16` technical domains from `2020-01-01` to `2025-12-31`. The design compares domains where early GenAI was a more plausible substitute.

We find reallocation inside a residualized public queue, not just decline. In more exposed domains, the remaining public questions become more context-heavy and less exposure-heavy. Within that queue, recent entrants become more visible in first-answer and early-endorsement roles than in accepted-answer certification roles. A certification-conversion ladder shows that first answers, first-positive answers, and top-scored answers are less likely to become accepted within the first week or month in more exposed domains after the transition. Tag-months with larger early-versus-accepted composition gaps exhibit slower endorsed resolution and weaker accepted-answer outcomes. In a strict question-side AI-ban timing test around `2022-12-05`, accepted-window certification weakens more clearly than immediate answer arrival. External and direct-AI validation layers are consistent with this interpretation.

The central implication is that private AI can leave a public platform active while residualizing the public queue, thinning incumbent answer capacity, reallocating visible answer work toward earlier roles, and weakening the link between early response and later public certification.

Keywords: generative AI; Stack Overflow; online knowledge communities; digital platforms; contributor composition; public answer supply; certification

## 1. Introduction

Generative AI can move technical problem solving out of public view without making a platform look empty. Some questions that once became public threads are now handled privately. What can change first is not visible activity, but who still does the public answering and whether those answers still become certified solutions.

This paper studies that shift on Stack Overflow. In domains where early GenAI was a more plausible substitute, the remaining posted queue becomes more context-heavy, newer contributors become more visible in early answer roles, and that work is less likely to become later certified resolution.

Stack Overflow is a public knowledge infrastructure, not just a stream of posts. Its value comes from a resolution process: someone answers, others endorse or correct, and one solution eventually becomes the reusable stopping point for later users. A platform can keep producing answers while that certification process weakens underneath.

We show two headline results and two interpretive layers. First, the remaining public queue in more exposed domains becomes more context-heavy and less exposure-heavy. Second, exposed domains become more recent-entrant-heavy in early and endorsed roles than in accepted resolution. When those role gaps widen, downstream certification weakens. Newly observed answerers in exposed domains are also less likely to return, though that durability result is supportive rather than central.

We do not treat the design as a single-date shock. Instead, we use the full `2025Q4` archive together with timing and validation checks to study how private AI shifts public answer work and later certification.

## 2. Literature and Theoretical Framing

### 2.1 Online Knowledge Communities as Public Problem-Solving Systems

Online knowledge communities create value not by generating posts per se, but by converting dispersed problem reports into reusable public knowledge. Prior work shows that this conversion is socially organized and disproportionately carried by experienced contributors who interpret ambiguous questions, supply legible answers, and help build a searchable archive (Wasko & Faraj 2005; Ma & Agarwal 2007; Butler et al. 2007; Bateman, Gray, & Butler 2011; Faraj, Jarvenpaa, & Majchrzak 2011).

This perspective makes contributor composition, not just contribution volume, central to platform performance. A Q&A site can remain visibly active while losing the contributors most able to turn questions into trusted solutions. When that happens, answer counts understate a deeper change in the platform's knowledge-production capacity.

That logic is especially relevant for Stack Overflow, where answering is shaped by reputation incentives, status dynamics, and core-periphery structure (Xu, Nian, & Cabral 2019; Smirnova, Reitzig, & Sorenson 2022; Safadi, Johnson, & Faraj 2020). A shift in who answers is therefore not merely a participation change; it is a change in the social organization of public problem solving.

### 2.2 Platform Coordination and Public Certification

A second step is to distinguish answering from resolution. In public Q&A systems, a thread can receive a response, attract visible endorsement, and reach a publicly recognized resolution, but these outcomes need not move together. They capture different coordination tasks: coverage of the question, community recognition of a plausible answer, and certification that the thread has produced a reusable stopping point for later searchers.

This distinction matters because platform change may appear first in the links between stages rather than in total activity. A system can remain responsive while becoming less able to convert responses into publicly certified solutions. Research on peer production and evaluation similarly shows that participation, evaluation, and control mechanisms jointly determine whether digital contributions become usable collective resources (Kane & Ransbotham 2016; Klapper, Piezunka, & Dahlander 2023; Mindel, Aaltonen, Rai, Mathiassen, & Jabr 2024). For this paper, the theoretical question is therefore whether early answer activity still translates into public certification.

### 2.3 Private AI as an Outside Option at the Public-Private Boundary

Generative AI introduces a private outside option for some kinds of technical problem solving. Users need not choose between ChatGPT and Stack Overflow in a strict binary sense; many will use both. The mechanism is marginal substitution: private AI can reduce the value of posting or answering some public questions, especially in domains where tasks are more local, snippet-solvable, templatable, or routine.

This outside-option view has two implications. First, some incumbent high-performing public contributors may answer less often because routine public work becomes less worthwhile once private AI handles more of it off-platform. Second, the barrier to posting a plausible public answer may fall for newer or less durable contributors. If both happen together, visible activity can persist even as the contributor mix changes and the connection between early response and later certification weakens.

Recent Stack Overflow work identifies adjacent effects of GenAI. Shan and Qiu (2025) find more answers after ChatGPT, but answers that are shorter and easier to read. Quinn and Gutt (2025) find that question asking falls, especially among casual users, while the remaining public queue becomes more complex and novel. What remains open is a supply-side platform question: conditional on questions still reaching the archive, who performs the visible answer work, where in the pipeline they appear, and how tightly early response remains linked to later public certification?

### 2.4 Three Distinct Mechanism Channels

The evidence cannot isolate a single mechanism, but it can separate three channels that are often blurred together.

The first is `incumbent exit`. Experienced public answerers may contribute less if routine public work becomes less worthwhile once private AI handles more of it off-platform. If this channel is important, the main governance problem is retention of durable contributors.

The second is `AI-augmented entry`. Newer or less established contributors may find it easier to post plausible answers when private AI lowers drafting and syntax costs. If this channel is important, the main governance problem is screening and certification rather than simple answer volume.

The third is `residual difficulty`. Routine questions may increasingly be resolved before they reach Stack Overflow, leaving a more context-specific and demanding public queue. If this channel is important, the issue is not only who answers, but what kinds of questions still require public resolution.

These channels imply different governance problems and should not be collapsed into one another. The data are strongest on the net reallocation they jointly produce. The durability results, role results, question-level residualization, and queue-composition evidence are most consistent with some mix of incumbent exit and AI-augmented entry operating against a harder residual queue. The evidence narrows the plausible account, but it does not fully separate the three forces.

### 2.5 Contributor Composition as a Capacity Indicator

The theoretical move is to treat contributor composition as a meaningful indicator of public resolution capacity. That claim has two parts. First, contributor composition is not just a background feature of the platform. Different contributor mixes imply different levels of response reliability, screening burden, and maintenance capacity. Second, the relevant composition is stage-specific, not only aggregate. It matters where in the answer pipeline different contributors appear. A platform that becomes more recent-entrant-heavy in first-response and top-score roles, but not to the same extent in accepted-current certification roles, operates differently from one in which all stages shift together.

This is also why the paper uses a bounded evidence hierarchy rather than one all-purpose regression. Question-level residualization and queue recomposition show how the public problem set changes. Role reallocation shows where newer public labor enters the answer pipeline. Certification consequences show why those role gaps matter for the archive. Durability and direct-observation layers then narrow plausible source stories without turning the paper into a mediation or clean-shock design.

### 2.6 Hypotheses

We translate that argument into four linked empirical families.

`H1. Residual public queue.` In more AI-substitutable domains, the remaining public queue should become less exposure-heavy and more context-heavy. If routine problems are increasingly handled privately, the marginal public question should look more residualized.

`H2. Answer-role mechanism.` If reallocation happens inside the public answer pipeline, it should be strongest in earlier and endorsed answer roles, not necessarily equally strong in accepted-current certification roles. The relevant contrast is not simply "entrants versus incumbents"; it is where in the answer pipeline recent entrants become more visible.

`H3. Certification consequences of role gaps.` If contributor composition is truly infrastructure-relevant, role-gap measures should predict downstream public-certification consequences. A tag-month in which recent entrants are more concentrated in early response than in accepted-current certification should display slower endorsed resolution or weaker public stopping signals.

`H4. Durability source margin.` In more AI-substitutable domains, the post-transition public entrant margin should become less durable. Newly observed answerers should be less likely to return and more likely to remain one-shot or low-repeat contributors.

These are directional mechanism hypotheses rather than an all-or-nothing chain, so the four families need not appear with equal strength. The current archival run is strongest on the queue-role-certification sequence, while durability remains supportive rather than coequal.

## 3. Setting, Data, Measures, and Empirical Design

### 3.1 Canonical Source Regime

The canonical source is the official Stack Overflow `2025Q4` dump. The analysis window runs from `2020-01-01` through `2025-12-31`. This source decision matters because the paper is about public contributor behavior, not about API snapshots or partial archival windows. Earlier source regimes are retained only as an audit trail for design chronology.

The long-window backbone contains `2,035,885` single-focal-tag questions and `2,391,883` matched answers across `16` focal technical domains:

- `android`
- `apache-spark`
- `bash`
- `docker`
- `excel`
- `firebase`
- `javascript`
- `kubernetes`
- `linux`
- `memory-management`
- `multithreading`
- `numpy`
- `pandas`
- `python`
- `regex`
- `sql`

This is the canonical evidence base for the `who still answers` paper. Older corrected-archive materials and API timing checks are no longer alternate paper identities. They are only provenance and limited validation.

The broader focal-question universe is larger. Across all questions carrying at least one focal tag, the `2025Q4` build contains `2,322,009` questions. The single-focal restriction therefore keeps `87.7%` of the focal-tag universe and excludes `12.3%` of multi-tag questions. The excluded questions carry more tags on average (`3.63` versus `3.24`), slightly more answers (`1.25` versus `1.17`), and a higher archive-date accepted rate (`0.514` versus `0.426`). This restriction therefore narrows the paper to cleaner exposure assignment rather than to the full Stack Overflow experience. The appendix reports the attrition table directly.

### 3.2 Focal Domains and Exposure Measure

The main explanatory variable is a continuous pre-shock exposure score at the tag level. The score is based on pre-shock domain language and intended to capture relative early AI substitutability, not direct AI use. Intuitively, the score rises with cues suggesting local, routine, snippet-solvable work and falls with cues suggesting environment dependence, hidden state, configuration complexity, or distributed context.

This design choice matters because the paper does not observe prompts, referrals, or user-level GenAI adoption. It instead asks whether public traces move differentially in domains where early private GenAI was more plausibly substitutable. The exposure variable is therefore a theory-driven proxy, and the claims should be read with that limit in mind.

The paper now reports the full tag ordering rather than asking readers to infer it indirectly. The five most exposed domains are `regex`, `pandas`, `numpy`, `sql`, and `apache-spark`. The five least exposed are `android`, `firebase`, `multithreading`, `kubernetes`, and `docker`, with `excel`, `javascript`, `bash`, `python`, and `linux` in the middle. That ordering is not beyond dispute, but it is face-valid enough to make the identifying contrast transparent, and the full ranking appears in the appendix.

The paper does not rely only on a hand-coded domain split to defend exposure. It also uses a question-level validation step built from `2048` labeled questions and `2304` audited API-assisted labels. The resulting continuous score achieves cross-validated `R^2 = 0.3779`, a cross-validated correlation of `0.6166`, and a correlation of `0.5097` with the legacy tag-level exposure index. More importantly, within-tag top-exposure questions remain more likely to receive rapid public response and accepted closure, while the predicted question-level exposure score itself falls in the high-exposure domains after the transition. This does not create complete direct AI-use measurement, but it makes the exposure proxy materially stronger than a frozen domain partition alone.

That distinction is important because it moves the paper closer to within-tag evidence. The strongest remaining criticism is not simply that exposure is proxied. It is that domain-level exposure alone may not show whether the residual public queue is actually changing. The question-level step directly addresses that concern. After the transition, the average question still reaching the legacy high-exposure queue becomes less exposure-heavy, while the higher-exposure questions that do remain continue to convert relatively well from rapid response into accepted closure. In short, the paper does not ask readers to trust a frozen domain split alone; it also shows question-level residualization inside the main setting.

The paper also adds external calibration on top of that internal validation. Using the official JetBrains Developer Ecosystem 2025 raw survey, the paper constructs five matched technical clusters spanning the focal domains: `SQL / analytics`, `Python / data`, `JavaScript / web`, `Android / mobile`, and `Shell / infra / cloud`. Across all five, the share of developers selecting `ChatGPT or other AI chatbots` as an answers platform exceeds the share selecting `Stack Overflow`, with a mean private-public gap of `0.077` and a largest gap of `0.098` in `JavaScript / web`. The appendix maps all `16` focal tags to these external clusters. This does not validate the exact exposure ordering mechanically, but it strengthens the substitution premise with independent developer-level microdata while keeping the exposure ranking transparent.

The main setting also carries a same-setting disclosed-AI dataset aligned to the canonical single-focal backbone. Using the official `2025Q4` dump, we scan question titles, question bodies, answer bodies, question comments, and answer comments for explicit AI-tool disclosures under a tightened pattern set designed to reduce false positives. The broad thread-level dataset identifies `9,147` focal questions (`0.4493%` of the canonical question backbone). A stricter disclosure subset identifies `2,407` threads, of which `1,932` are question-side disclosures visible at posting time and `575` are answer-side disclosures. This does not create complete direct-AI measurement in the main setting, but it narrows the "pure proxy" critique and it enables a cleaner ban-centered timing design on the question-side subset.

### 3.3 Sample Construction and Linked Objects

The paper uses four linked empirical objects that match the mechanism spine.

1. `Residual-queue panel`: question-level exposure and complexity measures test whether the remaining public queue becomes less exposure-heavy and more context-heavy.
2. `Answer-role panel`: for each question, the analysis identifies the `first_answer`, `first_positive`, `top_score`, and `accepted_current` role occupants.
3. `Bridge panel`: tag-months combine role-gap measures with downstream public-certification outcomes.
4. `Entrant durability panel`: newly observed focal-tag answerers are followed for `30`, `90`, `180`, and `365` days to measure return, one-shot status, and low-repeat behavior.

The residual-queue panel is question-centered and asks what kinds of problems still reach the archive. The role panel is also question-centered, but asks which kind of contributor occupies different slots within a thread's answer pipeline. The bridge panel is tag-month centered and asks whether composition gaps matter for downstream public certification once we aggregate across threads. The durability panel is contributor-centered: it follows answerers after their first observed focal-tag answer. These linked objects let the paper move from queue recomposition to role location to platform consequence, while treating durability as bounded evidence on the source margin of public labor.

Eligibility differs across these objects, and the paper states that explicitly. The durability panel includes newly observed answerers only when the forward-looking return window is observable. The role panel uses outcome-specific eligible questions: `first_answer` and `top_score` are defined on answered questions, `first_positive` is defined only where at least one answer eventually reaches a positive score, and `accepted_current` is defined only where the `2025Q4` dump records an accepted answer. The bridge panel aggregates only tag-months with both a role-gap numerator and an outcome-specific denominator. Bridge models are weighted by those denominators because the bridge is about conditional public certification once the relevant stage is reachable, not about mechanically conflating empty threads with certification failure.

### 3.4 Contributor and Role Definitions

The contributor constructs are intentionally narrow and frozen. The paper does not lead with a symmetric entrant-share result. Instead, it works with `recent entrants` and `newly observed public answerers` as contributor states inside the mechanism.

The most important role definitions are:

- `first_answer`: the earliest answer attached to a question
- `first_positive`: the earliest answer to obtain a positive score
- `top_score`: the answer with the highest score in the thread
- `accepted_current`: the answer occupying the accepted slot in the dump

These are not interchangeable roles. A contributor who supplies the first answer performs a queue-viability function. A contributor who supplies the first positive or top-scored answer performs an endorsement-related function. A contributor occupying the accepted-current role sits closest to the thread's visible certification endpoint. This is why the role family is substantively richer than a generic entrant-share coefficient.

### 3.5 Bridge and Complexity Measures

The core bridge variable is `recent_gap_first_vs_accepted`, defined as the difference between the recent-entrant share in the `first_answer` role and the recent-entrant share in the `accepted_current` role. A larger value means recent entrants are more visible in early response than in later accepted certification.

The main downstream outcomes are:

- `first_positive_answer_latency_mean`
- `accepted_cond_any_answer_30d_rate`
- `accepted_vote_30d_rate`

The main selection/composition proxies are:

- `body_word_count_mean`
- `tag_count_full_mean`
- `has_edit_mean`
- `residual_queue_complexity_index_mean`
- `complexity_index_broad_mean`

These measures are not claimed to be perfect reflections of hidden-state complexity. They are used as a second evidence chain. The paper is strongest when these proxies point in the same direction as the role and bridge families, not when every proxy is forced into significance.

### 3.6 Empirical Specifications

The primary role and durability estimands take the general form:

`y_it = alpha_i + gamma_t + theta_i t + beta(Exposure_i x Post_t) + X_it' lambda + epsilon_it`

where `i` indexes focal tags or contributor-tag objects, `t` indexes months or contributor-entry cohorts, `alpha_i` are fixed effects, `gamma_t` are month fixed effects, `theta_i t` are tag-specific linear trends when appropriate, and `X_it` includes queue-complexity controls in the complexity-controlled specifications.

For the bridge family, the key regressor is the role-gap variable itself:

`Certification_it = alpha_i + gamma_t + theta_i t + phi Gap_it + X_it' lambda + epsilon_it`

The bridge is therefore not tested as a pure exposure interaction. It is tested as a downstream consequence of the composition gap the paper argues is infrastructure-relevant.

The three families use different estimands. Durability models follow newly observed contributors over return windows. Answer-role models study who occupies different slots in the answer pipeline. Bridge models ask whether those role gaps predict later certification outcomes. The question-level identification layer is separate again: it uses within-tag exposure variation to test whether the residual public queue itself changes after the transition. Keeping these pieces separate avoids asking one regression to do too much.

### 3.7 Inferential Hierarchy and Timing Discipline

Because the tag universe is only `16`, few-cluster inference matters. The paper therefore reports conventional clustered p-values, CR2 small-sample corrections, wild-cluster bootstrap results, and randomization inference where appropriate. This is not decorative. It determines which results should carry the article.

The paper's mechanism spine is:

1. `residualized public queue`
2. `answer-role reallocation`
3. `certification consequences of role gaps`
4. `durability as a bounded source-margin layer`

The strict inferential hierarchy is not identical to the narrative hierarchy. Bridge outcomes remain strongest under CR2 and wild-cluster bootstrap, role outcomes survive the conservative inference suite on their preferred windows, and durability is directionally consistent but weaker under the strictest resampling and timing checks. We keep that distinction explicit so the paper does not pretend empirical symmetry while still leading with the same-setting mechanism stack.

The design is observational and differential, not a pristine natural experiment. We therefore describe the main setting as `ChatGPT-period differential exposure` rather than release-date treatment. That wording fits the diagnostics: key outcomes move in the post period, but many also accelerate before late 2022. The identifying value comes from consistent exposure-graded differences across a long archival window, not from one clean break. This interpretation is more consistent with modern difference-in-differences discipline, which emphasizes honest treatment of dynamics, serial correlation, and identifying assumptions rather than mechanical reliance on a single pre/post coefficient (Abadie 2005; Bertrand, Duflo, & Mullainathan 2004; Callaway & Sant'Anna 2021; Goodman-Bacon 2021; Rambachan & Roth 2023).

The pre-trend diagnostics are therefore not a footnote. In pre-shock slope tests, `brand_new_platform_share`, `first_answer_1d_rate_closure`, and `accepted_vote_30d_rate` all show meaningful differential pre-trends by exposure, while `residual_queue_complexity_index_mean` does not. This is exactly why the paper does not ask readers to treat the main design as a clean pre/post experiment. The broad window identifies a long-run exposure gradient. Narrower timing layers are used to test whether the same part of the pipeline moves when AI becomes more salient.

To go beyond a single post indicator, the paper triangulates timing in three ways. First, it uses an external monthly AI-salience series based on public `ChatGPT` Wikipedia pageviews. The timing question is then not only "before versus after late 2022." It is also "when public AI salience is higher, do more exposed domains exhibit larger role gaps or weaker downstream certification?" This series is not a treatment instrument, but it is a more disciplined timing check than a single post dummy. Because the tag universe remains `16`, this timing check is paired with permutation inference that reshuffles the tag-level exposure mapping rather than relying only on asymptotic clustered p-values.

Second, it uses a conservative same-setting internal title trace. After tightening the AI mention rules, the trace records `1,201` title hits across the canonical question backbone and correlates `0.794` with the external monthly `ChatGPT` salience series. Third, it uses a strict question-side AI-ban timing layer around the Stack Overflow AI-content ban date (`2022-12-05`). This layer keeps only disclosures visible when the question is posted and estimates narrow local windows with tag fixed effects, local linear trends, and donut sensitivity. It does not become a clean discontinuity design, but it improves the paper's ability to say something informative about a restricted timing margin.

Table 1 summarizes the data, outcomes, and inference plan.

| Table 1. Data, outcomes, and inference plan | Unit | Value | What it does here |
| --- | --- | ---: | --- |
| Questions | question rows | `2,035,885` | main Stack Overflow backbone |
| Answers | answer rows | `2,391,883` | answer-role and certification outcomes |
| Focal domains | tag universe | `16` | exposure contrast |
| Entrant durability | contributor-entry windows | `30/90/180/365d` | return and one-shot outcomes |
| Answer roles | role occupancy | `first_answer`, `first_positive`, `top_score`, `accepted_current` | who appears where |
| Certification bridge | tag-month outcomes | latency and certification | downstream consequence layer |
| Conservative inference | p-value suite | cluster, CR2, wild, RI | keeps the hierarchy honest |
| Timing checks | added layers | salience, title trace, ban window | narrows timing claims |

### 3.8 External Calibration and a Direct-AI-Use Second Setting

The paper now carries two external layers beyond the main Stack Overflow design. The first is the JetBrains calibration layer described above. Its function is not to identify causal effects inside the main setting. Its function is to show, with official developer-survey microdata, that the public-private answer-source substitution premise is not speculative. This is an exposure-calibration layer, not a replacement identification design.

The second is a direct-AI-use second setting built from the public `AIDev` dataset. `AIDev` explicitly labels AI-authored pull requests by tool family, including `OpenAI_Codex`, `Copilot`, `Cursor`, `Devin`, and `Claude_Code`. The upgraded version no longer stops at a generic overlap-repository panel. It also adds repository-domain overlap and fix-like task restrictions so that the second setting looks more like a public technical problem-solving environment rather than generic GitHub activity. This second setting is not a one-for-one replication of Stack Overflow, and the paper does not pretend otherwise. What it adds is a public technical collaboration environment in which direct AI use is observed rather than proxied and in which early visible feedback can again be compared with later certification outcomes.

The overlap rule is intentional. The baseline `AIDev` layer keeps repositories in which both AI-authored and human pull requests appear; the stricter version keeps only repository-month cells in which both appear. The upgraded version then narrows further to repository-domain families that overlap the Stack Overflow focal domains and to `fix`/`perf`/`refactor` tasks that are more plausibly question-like than generic feature delivery. Repository and month fixed effects then compare AI and human pull requests inside the same public review environment rather than across structurally different repositories. This gives up scale, but it makes the comparison more like the manuscript's object than the earlier generic prototype.

These external layers do different jobs. JetBrains calibrates the substitution premise. The same-setting disclosed-AI corpus and internal title trace tighten timing and partial direct observation inside the main environment. `AIDev` strengthens the external case that direct AI use can widen the gap between early visible response and later certification. No single layer has to do everything alone.

## 4. Results

### 4.1 Overview

The central empirical fact is reallocation inside a residualized public queue, not generic decline. In more exposed domains, the remaining public queue becomes less exposure-heavy and more context-heavy. Within that queue, recent entrants become more visible in early and endorsed answer roles than in accepted-current certification. The next question is whether those role gaps matter for the archive's certification function. They do: when early roles become more recent-entrant-heavy than accepted-current certification, downstream public certification weakens. Durability then helps explain why that newer public labor margin is less stable, but it is a bounded source-margin layer rather than a coequal pillar.

The paper should therefore be read as a bounded mechanism stack: `residualized public queue -> answer-pipeline reallocation -> certification consequences`, with durability and direct-observation layers used to narrow interpretation rather than define the article's identity.

Table 2 reports the main same-setting mechanism and consequence estimates.

| Table 2. Main same-setting mechanism and consequence estimates | Coefficient | Effect vs. mean outcome | Cluster p | CR2 p | Wild p | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Recent entrants in the first positive role | `+0.0281` | `+7.2%` | `0.0022` | `0.0108` | `0.0276` | entrant shift in endorsed early roles |
| Recent entrants in the top-score role | `+0.0272` | `+6.1%` | `0.0003` | `0.0042` | `0.0175` | entrant shift in top-scored roles |
| Expert incumbent mean answer activity | `-0.1978` | `-13.8%` | `0.0046` | n/a | n/a | thinner incumbent answer capacity |
| Acceptance lag for first answers that become accepted | `+2.60` days | `+17.4%` | `0.0016` | n/a | n/a | slower conversion into certified resolution |
| Residual queue complexity index | `+0.0551` | n/a | `0.00003` | n/a | n/a | more context-heavy posted queue |
| First-vs-accepted gap -> accepted votes in 30 days | `-0.0752` | `-36.1%` | `0.0019` | `0.0066` | `0.0075` | weaker public certification |
| First-vs-accepted gap -> time to first positive answer | `+283.9` minutes | `+127.5%` | `0.0039` | `0.0124` | `0.0100` | slower endorsed resolution |
| Return within 365 days | `-0.0266` | `-15.5%` | `0.0004` | `0.0047` | `0.0652` | less durable entrant margin |

The normalized-effect column is included only to help compare rows with different units. For rate outcomes it reports the coefficient relative to the mean outcome level. For latency it reports the implied change relative to mean latency.

### 4.2 Question-Level Residualization and Within-Tag Identification

The tag-level exposure split is useful, but it is coarse. We therefore ask a simpler within-tag question first: what does the public queue look like after the transition?

In the legacy high-exposure domains, the queue becomes less exposure-heavy. The share of top-exposure questions falls (`-0.0212`, clustered `p = 0.0840`). The average within-tag exposure score also declines (`-0.0656`, clustered `p = 0.0564`). These are not always threshold-clearing on their own. Taken together, they support the mechanism premise. The routine part of the old high-exposure queue is increasingly handled elsewhere.

Conditional on being posted, the remaining high-exposure questions are still relatively tractable. Within tags, the `post x top_quintile` coefficient is `+0.0197` for `first_answer_1d_rate`, `+0.0235` for `accepted_30d_rate`, and `+0.0167` for `accepted_given_fast_rate` (all with very small clustered p-values). The point is not that high-exposure questions became uniformly harder. It is that they are less likely to be the marginal public questions. The marginal public question is more context-heavy.

### 4.3 Residual Queue Recomposition

The mechanism story is incomplete without a queue story. We therefore check whether the remaining public queue becomes more context-heavy in more exposed domains.

The strongest complexity results are:

- `body_word_count_mean`: `+4.10`, `p = 0.0143`
- `tag_count_full_mean`: `+0.144`, `p = 0.00015`
- `residual_queue_complexity_index_mean`: `+0.0551`, `p = 0.00003`

The broader complexity composite also moves in the predicted direction. Code- and error-based proxies are noisier, and we do not lean on them as stand-alone evidence. The point is that multiple conceptually different proxies agree.

This evidence changes the interpretation. It is not only that answerers look different. The queue looks different. That makes the role and certification results easier to understand. When the queue is more context-heavy, it matters more who shows up early and who later anchors certification.

Table 3 summarizes the strongest queue-composition results.

| Table 3. Queue composition evidence | Coefficient | p-value | Interpretation |
| --- | ---: | ---: | --- |
| `body_word_count_mean` | `+4.10` | `0.0143` | longer question bodies |
| `tag_count_full_mean` | `+0.144` | `0.00015` | greater tag complexity |
| `has_edit_mean` | positive but weaker | secondary | more clarification work |
| `residual_queue_complexity_index_mean` | `+0.0551` | `0.00003` | strongest composite evidence |
| `complexity_index_broad_mean` | positive | supportive | noisy but aligned broad composite |

### 4.4 Mechanism Layer: Answer-Role Reallocation

The role family asks whether the entrant shift is concentrated in particular answer roles rather than simply in aggregate counts.

The strongest answer is yes. In the complexity-controlled role panel, `recent_entrant_90d_share` rises most clearly in:

- `first_answer`: `0.0245`, `p = 0.0017`
- `first_positive`: `0.0281`, `p = 0.0022`
- `top_score`: `0.0272`, `p = 0.0003`

The same coefficient is smaller and only marginal in `accepted_current`:

- `accepted_current`: `0.0169`, `p = 0.0617`

This is the key same-setting mechanism read. The paper does not claim that accepted-current certification is untouched. It claims that the entrant-heavy shift is strongest in rapid-response and endorsed-answer roles and weaker in the final accepted-current slot.

This role family is not simply a `brand_new_platform` story. Across roles, `brand_new_platform_share` does not produce a takeover pattern; in the decomposition table it is negative in first-answer, first-positive, top-score, and accepted-current roles. The more stable interpretation is thinner incumbent capacity and broader recent-entrant reallocation, not a surge of brand-new platform users. Expert incumbent mean answer activity falls in more exposed domains after the transition (`-0.1978`, `p = 0.0046`), and incumbent shares fall in the main answer roles, including `first_positive` (`-0.0600`, `p = 0.0140`) and `top_score` (`-0.0501`, `p = 0.0168`). That distinction matters because it prevents the mechanism from collapsing into a simplistic claim that "newcomers took over" the platform. It also clarifies the next step in the argument: if incumbent capacity thins while recent entrants become more visible in early and endorsed slots, later certification can become more weakly connected to the public labor arriving first.

The conservative inference stack is relatively favorable for the role family. For `first_positive`, the coefficient clears `CR2 p = 0.0108`, `wild p = 0.0276`, and `RI p = 0.0100`. For `top_score`, it clears `CR2 p = 0.0042`, `wild p = 0.0175`, and `RI p = 0.0080`. The formal pooled asymmetry term comparing `first_positive` to `accepted_current` is weaker (`CR2 p = 0.0963`, `wild p = 0.5188`, `RI p = 0.0341`). We therefore read this family as evidence about where the entrant shift is strongest, not as a perfectly sharp asymmetry result.

The construct-sensitivity battery reinforces that reading. The role family is weak on `30d` entrant windows, strongest on `90d`, and still present on `365d`. That pattern says the result is not driven by a hyper-short entrant definition. It also keeps us from claiming that every reasonable entrant window yields the same magnitude or precision.

### 4.5 Certification Consequences of Role Gaps

The consequence question is whether these role patterns matter for visible platform outcomes. The main bridge variable is `recent_gap_first_vs_accepted`: the difference between the recent-entrant share in the `first_answer` role and the recent-entrant share in the `accepted_current` role.

This variable is informative. Tag-months with a larger first-vs-accepted gap show:

- slower `first_positive_answer_latency_mean`: `283.9`, `p = 0.0039`
- weaker `accepted_cond_any_answer_30d_rate`: `-0.0704`, `p = 0.0509`
- weaker `accepted_vote_30d_rate`: `-0.0752`, `p = 0.0019`

These certification consequences are strongest on public stopping and ratification. In higher-complexity tag-months, the coefficient on `accepted_vote_30d_rate` becomes `-0.0928` (`p = 0.0019`). Leave-two-out stability is also unusually clean: the coefficient on the gap term stays negative in `100%` of dropped-tag pairs for both `accepted_vote_30d_rate` and `accepted_cond_any_answer_30d_rate`.

The role-level conversion ladder points in the same direction. For first answers, first-positive answers, and top-scored answers, early visible work becomes less likely to receive accepted certification within the first week or month in more exposed domains after the transition. The `accepted_30d` coefficient is `-0.0077` for first answers (`p = 0.0160`), `-0.0102` for first-positive answers (`p = 0.0331`), and `-0.0074` for top-scored answers (`p = 0.0251`). The `accepted_7d` coefficients are similarly negative for all three roles. Archive-level positive-score measures do not fall in the same way, so the result is not a generic collapse in endorsement. It is a narrower certification-conversion result: early visible work converts into accepted public resolution less often.

This is the strongest empirical result in the paper. It should nonetheless be read carefully. It is not mediation. The exposure coefficient is not eliminated across outcomes, and the paper does not observe every relevant screening or asker-ratification process. The right claim is narrower and stronger: a larger role gap between entrant-heavy first response and accepted-current certification is associated with weaker downstream public certification. That is enough to make contributor composition infrastructure-relevant rather than merely descriptive, because certification is the step that turns visible activity into a reusable archive.

The external timing extension makes this consequence family more persuasive rather than less. Using public monthly `ChatGPT` pageviews as an external AI-salience series, more exposed domains show a larger `recent_gap_first_vs_accepted` when external AI salience is higher (`0.0036`, clustered `p = 0.0123`, permutation `p = 0.0336`). The same external layer also predicts slower `first_positive_answer_latency_mean` (`+24.0` minutes, clustered `p = 0.0033`, permutation `p = 0.0138`) and weaker `accepted_cond_any_answer_30d_rate` (`-0.0032`, clustered `p = 0.0431`, permutation `p = 0.0856`). By contrast, the same extension does not generate a generic collapse result on `accepted_vote_30d_rate`, `any_answer_7d_rate`, or `first_answer_1d_rate_closure`. That specificity fits the mechanism stack. External AI salience appears to intensify the gap between rapid entrant-heavy response and later public certification rather than uniformly depressing every public outcome.

### 4.6 Bounded Source-Margin Evidence: Durability of Public Entrants

The durability family asks whether newly observed public answerers in exposed domains become less likely to convert into repeated public labor.

The answer is yes, though this layer is not as strong as the queue-role-certification stack. In the entrant-durability panel, more exposed domains show lower return probabilities and higher one-shot participation after the transition:

- `return_30d`: `-0.0217`, `p = 0.0006`
- `return_90d`: `-0.0239`, `p = 0.0005`
- `return_180d`: `-0.0256`, `p = 0.0004`
- `return_365d`: `-0.0266`, `p = 0.0004`
- `one_shot_365d`: `+0.0266`, `p = 0.0004`
- `low_repeat_90d`: `+0.0185`, `p = 0.0019`

Subtype splits show that the strongest `365`-day durability loss appears among `brand_new_platform` entrants (`-0.0369`, `p < 0.001`). This does not prove that every new entrant is low quality. It shows something narrower and more infrastructure-relevant: the post-transition public entrant margin in exposed domains is less durable.

`return_365d` yields `CR2 p = 0.0047`, `wild p = 0.0652`, and `RI p = 0.1242`. The right interpretation is therefore limited: durability clearly moves in the expected direction, but it is not the result that should carry the paper by itself.

### 4.7 Timing Validation: External Salience and Internal AI-Mention Trace

This is not a one-date shock design, so timing is a validation question. Do the main patterns strengthen when AI is more salient, in and out of the platform? For the role-gap and certification outcomes, the answer is yes.

First, we use monthly `ChatGPT` Wikipedia pageviews as an external salience series. We interact it with the tag-level exposure index inside the role-gap/certification panel, with tag and month fixed effects, tag-specific linear trends, and the residual queue complexity control. This is not an instrument. It is a timing check.

The most informative external-salience results are:

- `recent_gap_first_vs_accepted`: `+0.0036`, clustered `p = 0.0123`, permutation `p = 0.0336`
- `first_positive_answer_latency_mean`: `+24.0` minutes, clustered `p = 0.0033`, permutation `p = 0.0138`
- `accepted_cond_any_answer_30d_rate`: `-0.0032`, clustered `p = 0.0431`, permutation `p = 0.0856`

Second, we use a conservative internal trace. The platform itself starts naming AI tools. Under the tightened mention rules, there are `1,201` explicit AI-title hits across `2,035,885` focal questions, and that series correlates `0.794` with the external salience series. When we replace external salience with this internal trace, the same consequence outcomes still move in the expected direction: `recent_gap_first_vs_accepted` rises by `0.0060` (`cluster p = 0.0220`, permutation `p = 0.1554`) and `first_positive_answer_latency_mean` rises by `49.8` minutes (`cluster p = 0.0006`, permutation `p = 0.0140`).

Third, we use a strict question-side AI-ban timing layer around `2022-12-05`. This timing layer keeps only strict question-side disclosures visible when the question is posted. In high-exposure tags, accepted-window outcomes weaken more coherently than immediate answer arrival once the donut is applied. The cleanest promoted rows are:

- `accepted_30d`, `+/-45d`, donut `3d`: `-0.2633`, `p = 0.0408`
- `accepted_30d`, `+/-60d`, donut `3d`: `-0.2393`, `p = 0.0212`
- `first_answer_1d`, `+/-60d`, donut `3d`: `-0.0865`, `p = 0.4887`

These checks do not turn the paper into a clean discontinuity study. They support a narrower point. When AI is more salient, the same part of the public pipeline moves: the link between early response and accepted-window certification becomes weaker.

### 4.8 External Calibration and a Direct-AI-Use Second Setting

Two external sources speak directly to the remaining measurement objections.

First, we use JetBrains Developer Ecosystem 2025 microdata. The raw survey contains `24,534` respondents and measures both answer-source choice and AI-assisted coding use. In all five matched technical clusters, `ChatGPT or other AI chatbots` is chosen more often than `Stack Overflow` as an answers platform. The mean gap is `0.077`. The appendix maps each of the `16` focal tags to one of these clusters, so readers can see both the internal exposure rank and the external calibration surface. This does not validate our exact exposure ordering. It does validate the premise that private AI is already a serious substitute in the clusters that anchor our main setting.

Second, we use direct-AI validation settings that separate object fit from certification fit.

`DevGPT` (NAIST-SE 2023) provides a question-like troubleshooting environment: issues and discussions that explicitly share ChatGPT conversations. In this sample, `61.7%` of threads are updated within seven days, but only `34.3%` are closed within thirty days. The response-close gap is `0.274`. We use it as an object-fit check. It is selective and too thin to stand alone.

`AIDev` provides a stronger certification environment. In domain-overlap, fix-like work (`4,269` pull requests in `439` repositories), AI-authored pull requests get faster feedback and approval but weaker merge certification within thirty days (`-0.3201`, `p < 0.001`). The object is not Stack Overflow. The point is narrower: in a public technical setting with direct AI-use labels, early response and later certification can diverge in the same direction as in the main setting.

Table 4 summarizes the external evidence.

| Table 4. External calibration and direct-AI validation | Sample | Key estimate | What it adds |
| --- | --- | --- | --- |
| JetBrains 16-tag calibration | `24,534` developers, `5` matched clusters mapped to `16` focal tags | mean private-public answers-platform gap `= 0.077` | calibrates substitution premise |
| DevGPT question-like validation | `321` issue/discussion threads, `258` repos | `updated_7d - closed_30d = 0.274` | question-like direct-AI troubleshooting setting |
| AIDev certification validation | `4,269` fix-like PRs, `439` domain-overlap repos | `is_ai -> merged_30d = -0.3201`, `p < 0.001` | direct-AI certification check in a closer public setting |

### 4.9 Conservative Inference

The paper's core results survive conservative inference. They do not survive as a symmetric story in which everything moves equally strongly. That matters for interpretation.

The certification-consequence family is the cleanest. `bridge_accepted_vote_30d_gap` yields `CR2 p = 0.0066` and `wild p = 0.0075`. `bridge_first_positive_latency_gap` yields `CR2 p = 0.0124` and `wild p = 0.0100`. We do not report randomization inference for these role-gap terms because the object of interest is the composition gap itself.

The role family survives on the early and endorsed roles. It does not survive as a single sharp asymmetry result. Durability survives under clustered and CR2 inference, but weakens under the harshest resampling. It therefore belongs in the paper, but it is not the main claim.

Finally, timing must constrain the prose. Pre-period break diagnostics are mixed. This is not a clean release-date shock study. It is a long-window differential-exposure paper with a limited timing claim.

Table 5 summarizes the conservative gate.

| Table 5. Conservative inference checks | Cluster p | CR2 p | Wild p | RI p | How to read it |
| --- | ---: | ---: | ---: | ---: | --- |
| Return within 365 days | `0.0004` | `0.0047` | `0.0652` | `0.1242` | keep, but not a core result |
| One-shot participation within 365 days | `0.0004` | `0.0047` | `0.0802` | `0.0982` | keep, but not a core result |
| Recent entrants in the first positive role | `0.0022` | `0.0108` | `0.0276` | `0.0100` | main result |
| Recent entrants in the top-score role | `0.0003` | `0.0042` | `0.0175` | `0.0080` | main result |
| First-positive versus accepted asymmetry | `0.0423` | `0.0963` | `0.5188` | `0.0341` | secondary, not a core result |
| First-vs-accepted gap -> accepted votes in 30 days | `0.0019` | `0.0066` | `0.0075` | n/a | strongest family |
| First-vs-accepted gap -> time to first positive answer | `0.0039` | `0.0124` | `0.0100` | n/a | strongest family |

### 4.10 What Moved and What Did Not

Not everything moved, and that is part of the result.

What moved is consistent across the mechanism stack: the surviving public queue becomes more residualized, incumbent answer capacity thins, early and endorsed answer roles become more recent-entrant-heavy, certification takes longer within role-specific conversion ladders, and larger role gaps are associated with weaker downstream certification. Durability also weakens, but more modestly. What did not move is just as important. The sharp asymmetry term is not stable under the full conservative stack. Durability weakens under the harshest resampling. Accepted-current does not collapse cleanly, and the brand-new-platform margin does not support a simple newcomer-surge story.

This pattern fits reallocation, not platform failure. The public system stays active. The queue changes, visible labor shifts earlier in the pipeline, and the work that anchors certification does not move in parallel.

The external evidence is consistent with that reading. JetBrains supports the substitution premise. The same-setting disclosed-AI and ban-centered timing checks narrow the "pure proxy" critique. `DevGPT` adds a question-like direct-AI troubleshooting environment. `AIDev` adds a certification-focused direct-AI environment. None of these is a one-for-one replication. Together, they make a simple alternative story less convincing while leaving the design boundaries intact.

## 5. Discussion and Contribution

### 5.1 AI, Queue Residualization, and the Public-Private Reallocation of Problem-Solving Labor

This paper contributes to research on digital knowledge infrastructures by showing that generative AI can alter the public-private boundary of problem solving without producing an obvious collapse in public activity. The key implication is not simply that some users substitute from Stack Overflow to private AI. It is that private AI can change which tasks still reach the public archive and reorganize which contributors continue to perform visible answer work. A platform may therefore remain active in surface terms while becoming less effective at producing publicly reusable solutions.

### 5.2 Answer Roles and Platform Certification Capacity

The paper also contributes to platform governance research by arguing that stage-specific contributor composition is informative about a platform's certification capacity, not only its activity level. On Stack Overflow, first responses, community endorsement, and accepted-answer resolution serve distinct governance functions. When newer contributors provide a growing share of early responses but are less represented in later certification roles, the platform faces a different screening and validation problem even if answer volume remains high. On Stack Overflow, the reputation system rewards visible contribution, voting ranks candidate answers, and accepted-answer certification gives one solution thread-level closure. When recent entrants increasingly supply the first visible response but do not occupy certification roles at the same rate, the platform faces different screening, ranking, and validation problems than a platform in which those roles move together. Different contributor mixes therefore imply different onboarding costs, screening burdens, and certification risk, especially once the surviving public queue is more context-heavy.

### 5.3 Decomposing the Public Queue, the Answer Pipeline, and Certification

A third contribution is methodological. Prior work on Stack Overflow and generative AI has largely examined question activity, answer volume, or answer text. Those measures are useful, but they do not identify where change enters the production of public knowledge. By combining question-level residualization with first response, endorsed response, and accepted-answer certification, this paper distinguishes changes in the surviving public queue from changes in visible activity and from changes in publicly certified resolution. The empirical value of that distinction is that the platform can continue to generate answers while becoming less effective at converting those answers into trusted, reusable stopping points for later users.

External evidence supports the same reading. JetBrains supports the substitution premise. The internal title trace and the question-side ban window tie the mechanism to within-platform AI discourse and a narrow timing margin. `DevGPT` and `AIDev` show similar splits between response and certification in direct-AI settings.

### 5.4 Scope Conditions of the Claim

The claims remain narrow. The evidence does not identify clean expert exit, a single-date causal break, universal deterioration, or mediation. It also does not support a simple story in which brand-new users flood the answer pipeline. Instead, it establishes a bounded proposition: when private AI becomes a credible outside option, an archive-based platform can remain visibly active while the public queue residualizes, incumbent capacity thins, recent entrants become more visible in early roles, and visible effort converts more slowly and less reliably into certified public resolution. That is substantively important because the long-run value of a public knowledge archive depends not only on generating responses, but on leaving behind answers that later users can treat as credible and reusable.

## 6. Limitations, Policy Implications, and Future Research

### 6.1 Limitations

This study has four main limitations. First, the Stack Overflow setting does not permit complete observation of AI use. We observe selective thread-level disclosure, including `9,147` broad thread-level disclosures and `1,932` question-side disclosures visible at posting time, but disclosure remains an imperfect proxy for adoption. Second, timing remains mixed. The external salience series, the internal title trace, and the question-side ban window tighten the story, but they do not create a clean discontinuity. Third, the bridge analysis should be interpreted as evidence on downstream consequences rather than as a mediation test. Fourth, the external settings are complementary rather than symmetric replications. JetBrains calibrates the substitution premise. `DevGPT` is question-like but selective and thin. `AIDev` is certification-focused and directly labels AI use, but it is not Stack Overflow. Together these settings narrow major identification concerns without eliminating them.

### 6.2 Policy Implications

For platform governance, the central issue is not only whether expert contributors remain active, but also how answers are surfaced and certified when early response increasingly comes from recent entrants while accepted-answer certification remains associated with a different contributor mix. That pattern places pressure on onboarding, ranking, validation, and certification systems. Platforms should therefore monitor not only answer volume, but also the gap between who answers first and who ultimately supplies the answer the system certifies. Even without a full policy experiment, the implication is clear: generative-AI-era platform governance may depend less on blocking all AI-assisted participation than on absorbing, ranking, and certifying recent-entrant labor.

### 6.3 Future Research

Future work should sharpen mechanism measurement and test institutional responses more directly. Priorities include:

- question-level or session-level evidence on private AI substitution rather than domain-level exposure proxies
- richer enforcement or content-state traces that distinguish screening from ratification
- cross-platform evidence that more explicitly compares public Q&A and public code-review settings within a common certification framework
- platform-design interventions aimed at improving certification of recent-entrant answer supply

## 7. Conclusion

Private generative AI can change a public knowledge platform without obvious collapse. On Stack Overflow, the stronger signal is not a clean break in activity. It is a residualized public queue, a reallocation inside the public answer pipeline, and a weaker link between early response and later certification. Entrants become more visible in early and endorsed roles. Certification suffers most where first-response and accepted-current roles pull apart.

Private AI shifts the public-private boundary of problem solving. That shift changes who still does visible work and which answers become certified public resolution.

## References

Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of Economic Studies*, 72(1), 1-19.

Baird, A., & Maruping, L. M. (2021). The next generation of research on IS use: A theoretical framework of delegation to and from agentic IS artifacts. *MIS Quarterly*.

Bateman, P. J., Gray, P. H., & Butler, B. S. (2011). Research note: The impact of community commitment on participation in online communities. *Information Systems Research*, 22(4), 841-854.

Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1), 249-275.

Burtch, G., Lee, D., & Chen, L. (2024). Generative AI and participation decline in online technical communities. *Information Systems Research*.

Butler, B. S., Sproull, L., Kiesler, S., & Kraut, R. (2007). Community effort in online groups: Who does the work and why? In S. Weisband (Ed.), *Leadership at a Distance*. Lawrence Erlbaum.

Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

Cao, X., Duan, W., Gan, Q., & Whinston, A. B. (2023). Consequences of information feed integration on user engagement and contribution: A natural experiment in an online knowledge-sharing community. *Information Systems Research*.

Chen, J., Forman, C., & Kummer, M. E. (2025). Chat more and contribute better: An empirical study of a knowledge-sharing community. *Information Systems Research*.

Faraj, S., Jarvenpaa, S. L., & Majchrzak, A. (2011). Knowledge collaboration in online communities. *Organization Science*, 22(5), 1224-1239.

Forman, C., Ghose, A., & Wiesenfeld, B. (2008). Examining the relationship between reviews and sales: The role of reviewer identity. *Information Systems Research*, 19(3), 291-313.

Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

JetBrains. (2025). *Developer Ecosystem Survey 2025 raw data*. JetBrains Research.

Kane, G. C., & Ransbotham, S. (2016). Content as community regulator: The recursive performance effects of consumer contributions in online communities. *Organization Science*.

Klapper, H., Piezunka, H., & Dahlander, L. (2023). Peer evaluations: How ratings dynamics regulate the incentives of evaluators. *Organization Science*.

Li, H. (2026). *AIDev* [Data set]. Hugging Face.

Liang, H., Hong, Y., & Gu, B. (2024). Monitoring and the cold start problem in digital platforms: Theory and evidence from online labor markets. *Information Systems Research*.

Ma, M., & Agarwal, R. (2007). Through a glass darkly: Information technology design, identity verification, and knowledge contribution in online communities. *Information Systems Research*, 18(1), 42-67.

Majchrzak, A., Markus, M. L., & Wareham, J. (2016). Designing for digital transformation. *MIS Quarterly*, 40(2), 267-278.

Mindel, V., Aaltonen, A., Rai, A., Mathiassen, L., & Jabr, W. (2024). Timely quality problem resolution in peer-production systems. *Information Systems Research*.

NAIST-SE. (2023). *DevGPT (GPTShare)* [Data set]. GitHub / Zenodo.

Nickerson, J. V., & Zenger, T. R. (2004). A knowledge-based theory of the firm: The problem-solving perspective. *Organization Science*, 15(6), 617-632.

O'Mahony, S., & Ferraro, F. (2007). The emergence of governance in an open source community. *Academy of Management Journal*, 50(5), 1079-1106.

Quinn, B., & Gutt, D. (2025). Heterogeneous effects of generative artificial intelligence on knowledge seeking in online communities. *Journal of Management Information Systems*.

Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review of Economic Studies*, 90(5), 2555-2591.

Safadi, H., Johnson, S. L., & Faraj, S. (2020). Who contributes knowledge? Core-periphery tension in online innovation communities. *Organization Science*.

Smirnova, I., Reitzig, M., & Sorenson, O. (2022). Building status in an online community. *Organization Science*.

Tiwana, A. (2015). Evolutionary competition in platform ecosystems. *Information Systems Research*, 26(2), 266-281.

Wasko, M. M., & Faraj, S. (2005). Why should I share? Examining social capital and knowledge contribution in electronic networks of practice. *MIS Quarterly*, 29(1), 35-57.

Xu, Y., Nian, T., & Cabral, L. (2019). What makes geeks tick? A study of Stack Overflow careers. *Management Science*.

Shan, G., & Qiu, L. (2025). Examining the impact of generative AI on users' voluntary knowledge contribution: Evidence from a natural experiment on Stack Overflow. *Information Systems Research*.

Zhu, H., Kraut, R. E., & Kittur, A. (2015). The impact of membership overlap on the development of online communities. *Journal of Management Information Systems*, 31(4), 321-352.
