"""Streamlit frontend for the GA4GH Policy Compliance Chatbot.

Connects to the FastAPI backend (API_BASE_URL env var or localhost:8000).
Features:
- File upload (PDF, DOCX, TXT) or text paste
- Full analysis pipeline with progress bar
- Verdict banner (green/yellow/red)
- Gap report table with expandable details
- Per-gap remediation panel with copy button
- Follow-up chat
- PDF export
- RRF weight sidebar slider
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime

import httpx
import streamlit as st
from fpdf import FPDF

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300  # 5 minutes for full analysis

st.set_page_config(
    page_title="GA4GH Compliance Checker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ──────────────────────────────────────────────

for key, default in [
    ("gap_report", None),
    ("session_id", None),
    ("letter_text", ""),
    ("chat_history", []),
    ("analysis_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    rrf_weight = st.slider(
        "Lexical weight (RRF)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Balance between keyword (lexical) and semantic search. Default: 0.4 lexical / 0.6 semantic.",
    )

    st.divider()
    st.caption("**Models**")
    st.caption("Analysis: claude-sonnet-4-6")
    st.caption("Embeddings: text-embedding-3-large")

    st.divider()
    if st.session_state.analysis_done and st.session_state.gap_report:
        if st.button("📄 Export Report as PDF", use_container_width=True):
            pdf_bytes = _generate_pdf_report(st.session_state.gap_report)
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"ga4gh_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.divider()
    if st.session_state.analysis_done:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.gap_report = None
            st.session_state.session_id = None
            st.session_state.letter_text = ""
            st.session_state.chat_history = []
            st.session_state.analysis_done = False
            st.rerun()


# ── Helper functions ──────────────────────────────────────────────────────────


def _severity_badge(severity: str) -> str:
    """Return a colored badge string for a severity level."""
    colors = {"CRITICAL": "🔴", "MAJOR": "🟠", "MINOR": "🟡"}
    return colors.get(severity, "⚪")


def _match_badge(match_degree: str) -> str:
    """Return a status badge for a match degree."""
    badges = {
        "FULLY_MET": "✅ MET",
        "PARTIALLY_MET": "⚠️ PARTIAL",
        "NOT_MET": "❌ NOT MET",
    }
    return badges.get(match_degree, match_degree)


def _generate_pdf_report(gap_report: dict) -> bytes:
    """Generate a PDF compliance report from the gap report dict."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "GA4GH Policy Compliance Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 6, f"Session ID: {gap_report.get('session_id', 'N/A')}", ln=True, align="C")
    pdf.ln(5)

    # Verdict
    verdict = gap_report.get("verdict", "UNKNOWN")
    verdict_labels = {
        "VALID": "VALID - All critical requirements met",
        "INVALID_FIXABLE": "INVALID (Fixable) - Gaps found, remediation available",
        "INVALID_MAJOR_REVISION": "INVALID (Major Revision Required) - Critical gaps",
    }
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Overall Verdict: {verdict_labels.get(verdict, verdict)}", ln=True)
    pdf.ln(4)

    # Requirements Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Requirement Assessment Summary", ln=True)
    pdf.set_font("Helvetica", "", 9)

    for assessment in gap_report.get("assessments", []):
        req_id = assessment.get("requirement_id", "")
        description = assessment.get("description", "")[:60]
        severity = assessment.get("severity", "")
        match = assessment.get("match_degree", "")
        status_sym = {"FULLY_MET": "[MET]", "PARTIALLY_MET": "[PARTIAL]", "NOT_MET": "[NOT MET]"}.get(match, match)
        line = f"{req_id} [{severity}] {status_sym} - {description}"
        pdf.multi_cell(0, 6, line)

    pdf.ln(4)

    # Remediation section
    remediations = gap_report.get("remediations", [])
    if remediations:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Remediation Guidance", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for rem in remediations:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"{rem.get('gap_id')} - {rem.get('clause_category', '')}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, rem.get("explanation", "")[:300])
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, "Suggested text: " + rem.get("suggested_text", "")[:400])
            pdf.ln(3)

    return bytes(pdf.output())


def _run_analysis(letter_text: str, rrf_weight: float) -> dict | None:
    """Submit letter text to the API and return the gap report dict."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{API_BASE_URL}/analyze/text",
                json={"letter_text": letter_text, "rrf_lexical_weight": rrf_weight},
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.status_code} — {response.text[:200]}")
            return None
    except httpx.TimeoutException:
        st.error("Analysis timed out. The letter may be very long. Please try again.")
        return None
    except httpx.ConnectError:
        st.error(f"Cannot connect to the API at {API_BASE_URL}. Is the server running?")
        return None


def _upload_and_analyze(uploaded_file, rrf_weight: float) -> dict | None:
    """Upload a file to the API and return the gap report dict."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{API_BASE_URL}/analyze",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                data={"rrf_lexical_weight": str(rrf_weight)},
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.status_code} — {response.text[:200]}")
            return None
    except httpx.TimeoutException:
        st.error("Analysis timed out.")
        return None
    except httpx.ConnectError:
        st.error(f"Cannot connect to the API at {API_BASE_URL}.")
        return None


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🧬 GA4GH Policy Compliance Checker")
st.caption("Validate your data-use letter against GA4GH genomic data sharing policies.")

# ── Upload / input section ────────────────────────────────────────────────────

if not st.session_state.analysis_done:
    tab_file, tab_text = st.tabs(["📎 Upload File", "✏️ Paste Text"])

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload your data-use letter",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT",
        )
        if uploaded_file:
            st.success(f"File loaded: **{uploaded_file.name}** ({len(uploaded_file.getvalue()):,} bytes)")
            if st.button("🔍 Analyze Letter", key="btn_file", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Parsing document...")
                time.sleep(0.3)

                with st.spinner("Running compliance analysis..."):
                    progress_bar.progress(20, text="Chunking letter...")
                    time.sleep(0.2)
                    progress_bar.progress(40, text="Searching policy database...")
                    result = _upload_and_analyze(uploaded_file, rrf_weight)
                    progress_bar.progress(70, text="Analyzing compliance...")
                    time.sleep(0.3)
                    progress_bar.progress(90, text="Generating remediation...")
                    time.sleep(0.2)
                    progress_bar.progress(100, text="Complete!")

                if result:
                    st.session_state.gap_report = result.get("gap_report", {})
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.analysis_done = True
                    st.rerun()

    with tab_text:
        letter_text = st.text_area(
            "Paste your data-use letter here",
            height=350,
            placeholder="Dear Members of the Data Access Committee,\n\nI am writing to formally request...",
        )
        if st.button("🔍 Analyze Letter", key="btn_text", type="primary",
                     disabled=len(letter_text.strip()) < 50, use_container_width=True):
            progress_bar = st.progress(0, text="Parsing document...")

            with st.spinner("Running compliance analysis..."):
                progress_bar.progress(10, text="Chunking letter...")
                time.sleep(0.2)
                progress_bar.progress(30, text="Searching policy database...")
                result = _run_analysis(letter_text, rrf_weight)
                progress_bar.progress(70, text="Analyzing compliance...")
                time.sleep(0.3)
                progress_bar.progress(90, text="Generating remediation...")
                time.sleep(0.2)
                progress_bar.progress(100, text="Complete!")

            if result:
                st.session_state.gap_report = result.get("gap_report", {})
                st.session_state.session_id = result.get("session_id")
                st.session_state.letter_text = letter_text
                st.session_state.analysis_done = True
                st.rerun()

# ── Results section ───────────────────────────────────────────────────────────

if st.session_state.analysis_done and st.session_state.gap_report:
    report = st.session_state.gap_report
    verdict = report.get("verdict", "UNKNOWN")
    assessments = report.get("assessments", [])
    remediations = report.get("remediations", [])
    contradictions = report.get("contradictions", [])
    metadata = report.get("metadata", {})

    # ── Verdict banner ────────────────────────────────────────────────────────
    n_gaps = sum(1 for a in assessments if a.get("match_degree") != "FULLY_MET")
    n_not_met = sum(1 for a in assessments if a.get("match_degree") == "NOT_MET")

    if verdict == "VALID":
        st.success("✅ **VALID** — Your letter meets all critical requirements.")
    elif verdict == "INVALID_FIXABLE":
        st.warning(f"⚠️ **INVALID — Fixable** — {n_gaps} gaps found. Remediation is available below.")
    else:
        st.error(f"❌ **INVALID — Major Revision Needed** — {n_not_met} critical gap(s) must be addressed.")

    # Metadata pills
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requirements", "15")
    col2.metric("Fully Met", sum(1 for a in assessments if a.get("match_degree") == "FULLY_MET"))
    col3.metric("Partially Met", sum(1 for a in assessments if a.get("match_degree") == "PARTIALLY_MET"))
    col4.metric("Not Met", n_not_met)

    if metadata.get("elapsed_seconds"):
        st.caption(f"Analysis completed in {metadata['elapsed_seconds']}s | "
                   f"Session: {st.session_state.session_id}")

    st.divider()

    # ── Gap report table ──────────────────────────────────────────────────────
    st.subheader("📋 Requirement Assessment")

    for assessment in assessments:
        req_id = assessment.get("requirement_id", "")
        description = assessment.get("description", "")
        severity = assessment.get("severity", "MAJOR")
        match_degree = assessment.get("match_degree", "NOT_MET")
        evidence = assessment.get("evidence_from_letter", "")
        reasoning = assessment.get("reasoning", "")

        badge = _severity_badge(severity)
        status_badge = _match_badge(match_degree)
        header = f"{badge} **{req_id}** — {description[:70]}... — {status_badge}"

        with st.expander(header, expanded=(match_degree == "NOT_MET")):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("**Severity**")
                st.write(severity)
                st.caption("**Status**")
                st.write(match_degree.replace("_", " "))
            with col_b:
                st.caption("**Evidence from letter**")
                st.write(evidence if evidence != "No relevant content found" else "_Nothing found_")
            if reasoning:
                st.caption("**Reasoning**")
                st.write(reasoning)

    # ── Contradictions ────────────────────────────────────────────────────────
    if contradictions:
        st.divider()
        st.subheader("⚡ Detected Contradictions")
        for c in contradictions:
            severity = c.get("severity", "MINOR")
            with st.expander(f"{_severity_badge(severity)} Contradiction — {c.get('nature_of_contradiction', '')[:60]}"):
                st.caption("**Claim A**")
                st.write(c.get("claim_a", ""))
                st.caption("**Claim B**")
                st.write(c.get("claim_b", ""))
                st.caption("**Nature of Contradiction**")
                st.write(c.get("nature_of_contradiction", ""))

    # ── Remediation panel ─────────────────────────────────────────────────────
    if remediations:
        st.divider()
        st.subheader("🛠️ Remediation Guidance")
        st.caption("Suggested text you can add to your letter. Fields in **[BRACKETS]** need your input.")

        for rem in remediations:
            gap_id = rem.get("gap_id", "")
            severity = rem.get("severity", "MAJOR")
            clause = rem.get("clause_category", "")
            explanation = rem.get("explanation", "")
            suggested_text = rem.get("suggested_text", "")
            auto_fields = rem.get("auto_filled_fields", {})
            manual_fields = rem.get("manual_fields", [])

            with st.expander(
                f"{_severity_badge(severity)} **{gap_id}** — {clause}",
                expanded=(severity == "CRITICAL"),
            ):
                if explanation:
                    st.info(explanation)

                if auto_fields:
                    st.caption("**Auto-filled from your letter:**")
                    for field, value in auto_fields.items():
                        st.write(f"• **{field}:** {value}")

                if manual_fields:
                    st.caption("**You need to provide:**")
                    for mf in manual_fields:
                        field_name = mf.get("field_name", "") if isinstance(mf, dict) else mf
                        example = mf.get("example", "") if isinstance(mf, dict) else ""
                        st.write(f"• **{field_name}** _(e.g., {example})_")

                if suggested_text:
                    st.caption("**Suggested text to add:**")
                    st.code(suggested_text, language=None)
                    st.button(
                        "📋 Copy to clipboard",
                        key=f"copy_{gap_id}",
                        help="Click to copy the suggested text",
                        on_click=lambda t=suggested_text: st.write(
                            f'<script>navigator.clipboard.writeText({json.dumps(t)})</script>',
                            unsafe_allow_html=True,
                        ),
                    )

    # ── Follow-up chat ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Follow-up Questions")

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Ask about a specific gap, requirement, or how to improve your letter..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    with httpx.Client(timeout=60) as client:
                        resp = client.post(
                            f"{API_BASE_URL}/followup",
                            json={
                                "session_id": st.session_state.session_id,
                                "message": prompt,
                            },
                        )
                    if resp.status_code == 200:
                        answer = resp.json().get("response", "")
                    else:
                        answer = f"Error: {resp.status_code}"
                except Exception as e:
                    answer = f"Connection error: {e}"

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
