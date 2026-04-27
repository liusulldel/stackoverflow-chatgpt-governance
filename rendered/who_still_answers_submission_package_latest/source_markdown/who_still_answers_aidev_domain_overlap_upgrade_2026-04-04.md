# Who Still Answers: AIDev Domain-Overlap Upgrade

Date: April 4, 2026

## Why This Upgrade Exists

The baseline AIDev prototype established direct AI-use observation, but it still looked like a generic public-code-review analogue.
This upgrade pushes the second setting toward a more question-like and domain-aligned pillar by adding repository-domain overlap and fix-like task restrictions.

## Sample ladder

- `overlap_full`: `16,395` PRs, `816` repos, AI share `0.598`, fix-like share `0.386`
- `domain_overlap`: `11,542` PRs, `543` repos, AI share `0.594`, fix-like share `0.370`
- `domain_overlap_fixlike`: `4,269` PRs, `439` repos, AI share `0.646`, fix-like share `1.000`
- `strict_domain_overlap_fixlike`: `2,764` PRs, `253` repos, AI share `0.714`, fix-like share `1.000`

## Fixed-effect results

- `overlap_full`, `first_feedback_7d` on `is_ai`: coef `0.8267`, clustered `p = 0.0000`
- `overlap_full`, `approved_30d` on `is_ai`: coef `0.2414`, clustered `p = 0.0000`
- `overlap_full`, `merged_30d` on `is_ai`: coef `-0.2237`, clustered `p = 0.0000`
- `overlap_full`, `review_merge_gap` on `is_ai`: coef `1.0503`, clustered `p = 0.0000`
- `domain_overlap`, `first_feedback_7d` on `is_ai`: coef `0.8283`, clustered `p = 0.0000`
- `domain_overlap`, `approved_30d` on `is_ai`: coef `0.2480`, clustered `p = 0.0000`
- `domain_overlap`, `merged_30d` on `is_ai`: coef `-0.2455`, clustered `p = 0.0000`
- `domain_overlap`, `review_merge_gap` on `is_ai`: coef `1.0738`, clustered `p = 0.0000`
- `domain_overlap_fixlike`, `first_feedback_7d` on `is_ai`: coef `0.8517`, clustered `p = 0.0000`
- `domain_overlap_fixlike`, `approved_30d` on `is_ai`: coef `0.2591`, clustered `p = 0.0000`
- `domain_overlap_fixlike`, `merged_30d` on `is_ai`: coef `-0.3201`, clustered `p = 0.0000`
- `domain_overlap_fixlike`, `review_merge_gap` on `is_ai`: coef `1.1717`, clustered `p = 0.0000`
- `strict_domain_overlap_fixlike`, `first_feedback_7d` on `is_ai`: coef `0.8763`, clustered `p = 0.0000`
- `strict_domain_overlap_fixlike`, `approved_30d` on `is_ai`: coef `0.2557`, clustered `p = 0.0000`
- `strict_domain_overlap_fixlike`, `merged_30d` on `is_ai`: coef `-0.3354`, clustered `p = 0.0000`
- `strict_domain_overlap_fixlike`, `review_merge_gap` on `is_ai`: coef `1.2118`, clustered `p = 0.0000`

## Safe read

This still does not create a one-for-one Stack Overflow replication.
What it does add is a direct-AI-use public technical collaboration setting that is more domain-aligned and more fix-task-oriented than the earlier generic overlap-repo build.
If early feedback remains faster while later merge certification weakens inside the domain-overlap fix-like subset, the second pillar becomes much harder to dismiss as off-object.
