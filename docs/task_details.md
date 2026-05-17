Title: SSD-2026: Social Support Detection in Social Media

Goal: Build and evaluate NLP systems that (i) detect social support in social-media comments and (ii) identify the target of that support (individual vs. community and the specific community).

Motivation: Move beyond sentiment analysis toward actionable prosocial language understanding, enabling realistic end‑to‑end evaluation for deployment scenarios.

Subtask 1: Support Detection (Binary)
Input: A single social-media comment

Output: Support / Not Support

Definition: The comment expresses encouragement, care, admiration, help, or solidarity toward someone.

Subtask 2: Target Type Identification (Binary)
Input: Supportive comments

Output: Individual / Group

Definition: Determine whether the support is aimed at a specific person or a collective entity.

Subtask 3: Targeted Group Classification (Multiclass)
Input: Supportive comments targeting a group

Output classes: Nation, Other, LGBTQ, Black Community, Religion, Women

Definition: Identify which community is being supported.

Scope constraint: Supportive comments that explicitly promote violence or harm are labeled as Not Support.