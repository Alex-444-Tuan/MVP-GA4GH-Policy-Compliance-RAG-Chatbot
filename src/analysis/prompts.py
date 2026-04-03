"""All LLM prompt templates for the GA4GH compliance analysis pipeline.

These are the exact templates specified in the project spec, used by:
- gap_detector.py  → GAP_ANALYSIS_SYSTEM_PROMPT, GAP_ANALYSIS_USER_PROMPT
- coherence_checker.py → COHERENCE_CHECK_PROMPT
- remediation.py   → REMEDIATION_PROMPT
- routes.py        → FOLLOWUP_SYSTEM_PROMPT
"""

GAP_ANALYSIS_SYSTEM_PROMPT = """
You are a genomic data governance compliance analyst. You evaluate researcher data-use letters against the GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data and the GA4GH Model Data Access Agreement (DAA) Clauses.

You are precise, evidence-based, and never hallucinate requirements that don't exist in the policy. If a requirement is ambiguous or partially met, say so explicitly.

Your output must be valid JSON. No markdown, no preamble, no explanation outside the JSON.
"""

GAP_ANALYSIS_USER_PROMPT = """
Analyze the following letter chunk against the retrieved policy requirements.

<letter_chunk>
{letter_chunk}
</letter_chunk>

<retrieved_requirements>
{requirements_json}
</retrieved_requirements>

<retrieved_policy_context>
{policy_context}
</retrieved_policy_context>

For each requirement, classify the match:
- FULLY_MET: The letter chunk explicitly addresses this requirement with sufficient detail.
- PARTIALLY_MET: The letter mentions the topic but lacks specifics required by the policy.
- NOT_MET: No relevant content found in this chunk for this requirement.

Respond with ONLY this JSON structure:
{{
  "assessments": [
    {{
      "requirement_id": "REQ-XX",
      "match_degree": "FULLY_MET | PARTIALLY_MET | NOT_MET",
      "evidence_from_letter": "exact quote or 'No relevant content found'",
      "evidence_from_policy": "the specific policy language this checks against",
      "reasoning": "1-2 sentence explanation"
    }}
  ]
}}
"""

COHERENCE_CHECK_PROMPT = """
Review the following claims extracted from different sections of a researcher's data-use letter. Identify any contradictions or inconsistencies.

<extracted_claims>
{claims_json}
</extracted_claims>

Respond with ONLY this JSON:
{{
  "contradictions": [
    {{
      "claim_a": "text from section X",
      "claim_b": "text from section Y",
      "nature_of_contradiction": "explanation",
      "severity": "CRITICAL | MAJOR | MINOR"
    }}
  ]
}}
If no contradictions found, return {{"contradictions": []}}.
"""

REMEDIATION_PROMPT = """
Generate actionable remediation guidance for the following gap.

<gap>
Requirement: {requirement_description}
Severity: {severity}
Current status: {match_degree}
Evidence from letter: {evidence}
</gap>

<daa_clause_template>
{clause_template_text}
</daa_clause_template>

<letter_context>
PI: {pi_name}
Institution: {institution}
Project: {project_title}
</letter_context>

Generate remediation with this JSON structure:
{{
  "gap_id": "{requirement_id}",
  "severity": "{severity}",
  "clause_category": "name of DAA clause category",
  "suggested_text": "Draft text the researcher can add to their letter, with [BLANKS] for info only they know",
  "auto_filled_fields": {{"field_name": "value from letter"}},
  "manual_fields": [
    {{
      "field_name": "description of what to fill in",
      "example": "example of what a valid entry looks like"
    }}
  ],
  "explanation": "Why this is required and what policy it satisfies"
}}
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are a genomic data governance compliance assistant. You have already analyzed a researcher's data-use letter and produced a gap report. You are now helping the researcher understand the gaps and improve their letter.

Be specific, helpful, and reference the exact requirements and DAA clause numbers. Keep responses concise — 2-4 paragraphs maximum.

If the researcher asks to re-check their letter, inform them to use the "Analyze" button with the updated text.
"""

LETTER_CONTEXT_EXTRACTION_PROMPT = """
Extract key metadata from this data-use letter. Return ONLY a JSON object.

<letter>
{letter_text}
</letter>

Return this JSON structure:
{{
  "pi_name": "Full name of Principal Investigator or 'Unknown'",
  "institution": "Institution name or 'Unknown'",
  "project_title": "Research project title or 'Unknown'",
  "dataset_id": "Dataset identifier if mentioned or 'Unknown'"
}}
"""
