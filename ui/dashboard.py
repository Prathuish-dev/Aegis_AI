"""
Aegis AI — Streamlit Dashboard
================================
A real-time self-healing monitoring dashboard that connects to the
Aegis AI FastAPI backend.

Pages
-----
- 🏠 Live Incident Feed    — auto-refreshing table of all incidents
- 🔍 Incident Detail       — drill-down view with CoT steps & root cause
- 🚨 Manual Trigger        — submit a new anomaly report via the API
- ✅ HITL Approval Panel   — review and approve pending autonomous fixes
- 📊 Health Metrics        — service health, uptime and quick stats

Run with:
    streamlit run ui/dashboard.py
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
REFRESH_INTERVAL: int = 5  # seconds

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

STATUS_COLORS: Dict[str, str] = {
    "pending": "#FFA500",
    "pending_approval": "#FF4500",
    "suggested": "#1E90FF",
    "success": "#2ECC71",
    "failed": "#E74C3C",
}

CATEGORY_ICONS: Dict[str, str] = {
    "data_issue": "🗂️",
    "prompt_issue": "📝",
    "model_issue": "🤖",
    "system_issue": "⚙️",
    "incident_issue": "🚨",
}


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#888888")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;">{status.upper()}</span>'


def _category_icon(category: str) -> str:
    return CATEGORY_ICONS.get(category, "❓")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: Optional[Dict] = None) -> Optional[Any]:
    """GET request to the Aegis API. Returns parsed JSON or None on error."""
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params or {}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Aegis API. Is it running at `" + API_BASE_URL + "`?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(path: str, payload: Dict) -> Optional[Any]:
    """POST request to the Aegis API. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Aegis API.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page: Live Incident Feed
# ---------------------------------------------------------------------------

def page_incident_feed() -> None:
    st.title("🏠 Live Incident Feed")
    st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s. Data from `{API_BASE_URL}/incidents`.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_system = st.text_input("Filter by system_id", placeholder="e.g. llm-prod-01")
    with col2:
        filter_status = st.selectbox(
            "Filter by status",
            ["(all)", "pending", "pending_approval", "suggested", "success", "failed"],
        )
    with col3:
        page_size = st.number_input("Page size", min_value=5, max_value=100, value=20)

    params: Dict[str, Any] = {"page": 1, "page_size": page_size}
    if filter_system:
        params["system_id"] = filter_system
    if filter_status != "(all)":
        params["fix_status"] = filter_status

    data = _api_get("/incidents", params)
    if data is None:
        return

    incidents: List[Dict] = data.get("incidents", [])
    total: int = data.get("total", 0)

    st.markdown(f"**{total} total incidents** | Showing page 1 of {max(1, (total + page_size - 1) // page_size)}")

    if not incidents:
        st.info("No incidents found. Submit an anomaly via **Manual Trigger** to get started.")
        return

    # Build display table
    rows = []
    for inc in incidents:
        rows.append({
            "ID": inc["id"],
            "System": inc["system_id"],
            "Category": f"{_category_icon(inc['failure_category'])} {inc['failure_category']}",
            "Confidence": f"{inc.get('confidence_score', 0) * 100:.1f}%",
            "Status": inc["fix_status"],
            "Human Review": "⚠️ Yes" if inc.get("requires_human_review") else "✅ No",
            "Created": inc["created_at"][:19].replace("T", " "),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("🔍 Use **Incident Detail** in the sidebar to inspect a specific incident.")

    # Auto-refresh countdown
    placeholder = st.empty()
    for i in range(REFRESH_INTERVAL, 0, -1):
        placeholder.caption(f"⏱ Refreshing in {i}s…")
        time.sleep(1)
    st.rerun()


# ---------------------------------------------------------------------------
# Page: Incident Detail
# ---------------------------------------------------------------------------

def page_incident_detail() -> None:
    st.title("🔍 Incident Detail")

    incident_id = st.number_input("Enter Incident ID", min_value=1, step=1, value=1)
    if st.button("🔎 Load Incident", type="primary"):
        data = _api_get(f"/incidents/{incident_id}")
        if data is None:
            return
        st.session_state["detail_incident"] = data

    if "detail_incident" not in st.session_state:
        st.info("Enter an incident ID and click **Load Incident** to see the details.")
        return

    inc = st.session_state["detail_incident"]

    # Header
    icon = _category_icon(inc["failure_category"])
    st.subheader(f"{icon} Incident #{inc['id']} — {inc['system_id']}")
    st.markdown(
        _status_badge(inc["fix_status"]) + f" &nbsp; {'⚠️ Requires Human Review' if inc.get('requires_human_review') else ''}",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence Score", f"{(inc.get('confidence_score') or 0) * 100:.1f}%")
        st.metric("Fix Status", inc["fix_status"])
    with col2:
        st.metric("Failure Category", inc["failure_category"])
        st.metric("Human Review Required", "Yes" if inc.get("requires_human_review") else "No")

    st.markdown("### 📋 Anomaly Description")
    st.info(inc["anomaly_description"])

    st.markdown("### 🔬 Root Cause")
    st.warning(inc.get("root_cause") or "_No root cause diagnosed yet._")

    st.markdown("### 💡 Fix Recommendation")
    st.success(inc.get("fix_recommendation") or "_No recommendation available yet._")

    st.markdown("### 📦 Raw Metrics")
    st.json(inc.get("raw_metrics", {}))

    st.caption(f"Created: {inc['created_at']} | Updated: {inc.get('updated_at') or 'N/A'}")


# ---------------------------------------------------------------------------
# Page: Manual Trigger
# ---------------------------------------------------------------------------

def page_manual_trigger() -> None:
    st.title("🚨 Manual Anomaly Trigger")
    st.caption("Submit an anomaly directly to the Aegis AI pipeline for diagnosis.")

    with st.form("anomaly_form"):
        system_id = st.text_input("System ID *", placeholder="e.g. llm-prod-01")
        anomaly_description = st.text_area(
            "Anomaly Description *",
            placeholder="Describe what went wrong — accuracy drop, latency spike, hallucination burst…",
            height=120,
        )
        failure_category = st.selectbox(
            "Failure Category *",
            ["model_issue", "data_issue", "prompt_issue", "system_issue", "incident_issue"],
        )
        st.markdown("**Raw Metrics** (key=value, one per line)")
        raw_metrics_text = st.text_area(
            "Raw Metrics",
            placeholder="accuracy=0.71\nlatency_ms=520\nerror_rate=0.03",
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀 Submit Anomaly Report", type="primary")

    if submitted:
        if not system_id or not anomaly_description:
            st.error("System ID and Anomaly Description are required.")
            return

        # Parse raw metrics
        raw_metrics: Dict[str, Any] = {}
        for line in raw_metrics_text.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    raw_metrics[k.strip()] = float(v.strip())
                except ValueError:
                    raw_metrics[k.strip()] = v.strip()

        payload = {
            "system_id": system_id,
            "anomaly_description": anomaly_description,
            "failure_category": failure_category,
            "raw_metrics": raw_metrics,
        }

        with st.spinner("🤖 Aegis AI is diagnosing the anomaly…"):
            result = _api_post("/report-anomaly", payload)

        if result:
            st.success(f"✅ Incident #{result['id']} created successfully!")
            st.markdown("### Diagnosis Result")
            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{(result.get('confidence_score') or 0) * 100:.1f}%")
            col2.metric("Fix Status", result["fix_status"])
            st.markdown(f"**Root Cause:** {result.get('root_cause', 'N/A')}")
            st.markdown(f"**Recommendation:** {result.get('fix_recommendation', 'N/A')}")
            if result.get("requires_human_review"):
                st.warning("⚠️ This incident requires human review. Go to the **HITL Approval Panel**.")


# ---------------------------------------------------------------------------
# Page: HITL Approval Panel
# ---------------------------------------------------------------------------

def page_hitl_approval() -> None:
    st.title("✅ HITL Approval Panel")
    st.caption("Review and approve autonomous fix recommendations that require human sign-off.")

    data = _api_get("/incidents", {"page": 1, "page_size": 50, "fix_status": "pending_approval"})
    if data is None:
        return

    pending: List[Dict] = data.get("incidents", [])

    if not pending:
        st.success("🎉 No incidents pending approval. The system is running autonomously.")
        return

    st.warning(f"⚠️ **{len(pending)} incident(s) awaiting your review**")

    for inc in pending:
        icon = _category_icon(inc["failure_category"])
        with st.expander(
            f"{icon} Incident #{inc['id']} — {inc['system_id']} | "
            f"Confidence: {(inc.get('confidence_score') or 0) * 100:.1f}%",
            expanded=True,
        ):
            st.markdown(f"**Anomaly:** {inc['anomaly_description']}")
            st.markdown(f"**Root Cause:** {inc.get('root_cause') or 'N/A'}")
            st.info(f"💡 **Recommended Fix:** {inc.get('fix_recommendation') or 'N/A'}")
            st.json(inc.get("raw_metrics", {}))

            col1, col2 = st.columns(2)
            note = st.text_input(
                "Optional reviewer note",
                key=f"note_{inc['id']}",
                placeholder="Reason for approval or rejection…",
            )
            with col1:
                if st.button(f"✅ Approve #{inc['id']}", key=f"approve_{inc['id']}", type="primary"):
                    result = _api_post(
                        f"/incidents/{inc['id']}/approve",
                        {"approved": True, "approver_note": note},
                    )
                    if result:
                        st.success(f"Fix approved and executed. Status: **{result['fix_status']}**")
                        st.rerun()
            with col2:
                if st.button(f"❌ Decline #{inc['id']}", key=f"decline_{inc['id']}"):
                    result = _api_post(
                        f"/incidents/{inc['id']}/approve",
                        {"approved": False, "approver_note": note},
                    )
                    if result:
                        st.info(f"Fix declined. Status reverted to: **{result['fix_status']}**")
                        st.rerun()


# ---------------------------------------------------------------------------
# Page: Health Metrics
# ---------------------------------------------------------------------------

def page_health_metrics() -> None:
    st.title("📊 Health Metrics")

    health = _api_get("/health")
    if health is None:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API Status", "🟢 " + health.get("status", "unknown").upper())
    col2.metric("Version", health.get("version", "N/A"))
    col3.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    col4.metric("Total Incidents", health.get("total_incidents", 0))

    st.metric("Pending Approvals", health.get("pending_approvals", 0))

    st.markdown("---")

    # Incident status distribution
    all_data = _api_get("/incidents", {"page": 1, "page_size": 100})
    if all_data and all_data.get("incidents"):
        incidents = all_data["incidents"]
        import pandas as pd

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for inc in incidents:
            s = inc.get("fix_status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        st.markdown("### 🗂️ Incident Status Breakdown")
        status_df = pd.DataFrame(
            [{"Status": k, "Count": v} for k, v in status_counts.items()]
        )
        st.bar_chart(status_df.set_index("Status"))

        # Category breakdown
        cat_counts: Dict[str, int] = {}
        for inc in incidents:
            c = inc.get("failure_category", "unknown")
            cat_counts[c] = cat_counts.get(c, 0) + 1

        st.markdown("### 🏷️ Failure Category Breakdown")
        cat_df = pd.DataFrame(
            [{"Category": k, "Count": v} for k, v in cat_counts.items()]
        )
        st.bar_chart(cat_df.set_index("Category"))

        # Confidence score trend
        st.markdown("### 📈 Agent Confidence Score Trend")
        conf_df = pd.DataFrame([
            {
                "Incident": f"#{inc['id']}",
                "Confidence": (inc.get("confidence_score") or 0) * 100,
            }
            for inc in reversed(incidents)
        ])
        st.line_chart(conf_df.set_index("Incident"))
    else:
        st.info("Submit some anomaly reports to see metric charts here.")


# ---------------------------------------------------------------------------
# Main app entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Aegis AI Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.image(
        "https://img.shields.io/badge/Aegis%20AI-Self%20Healing-blue?style=for-the-badge&logo=robot",
        use_container_width=True,
    )
    st.sidebar.title("🛡️ Aegis AI")
    st.sidebar.caption("LLM Self-Healing Monitor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Live Incident Feed",
            "🔍 Incident Detail",
            "🚨 Manual Trigger",
            "✅ HITL Approval Panel",
            "📊 Health Metrics",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"API: `{API_BASE_URL}`")

    # Route to selected page
    if page == "🏠 Live Incident Feed":
        page_incident_feed()
    elif page == "🔍 Incident Detail":
        page_incident_detail()
    elif page == "🚨 Manual Trigger":
        page_manual_trigger()
    elif page == "✅ HITL Approval Panel":
        page_hitl_approval()
    elif page == "📊 Health Metrics":
        page_health_metrics()


if __name__ == "__main__":
    main()
