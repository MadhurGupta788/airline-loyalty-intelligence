import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide"
)

# ── Load data ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
PROC = BASE / "data" / "processed"

@st.cache_data
def load_data():
    df = pd.read_csv(PROC / "master_features.csv")
    return df

master = load_data()

# ── Colour palette ──────────────────────────────────────────────────────────
SEG_COLORS = {
    "Loyal Frequent Fliers" : "#1D9E75",
    "New Engagers II"       : "#378ADD",
    "Dormant High-Value"    : "#EF9F27",
    "At-Risk Inactives"     : "#D85A30",
    "New Engagers"          : "#7F77DD",
}
RISK_COLORS = {
    "Low"      : "#1D9E75",
    "Medium"   : "#EF9F27",
    "High"     : "#D85A30",
    "Critical" : "#8B0000",
}

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Airplane_silhouette.svg/200px-Airplane_silhouette.svg.png", width=60)
st.sidebar.title("✈️ Loyalty Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Member Lookup", "👥 Segment Explorer"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Total members: {len(master):,}")
st.sidebar.caption("Data window: 2017–2018")
st.sidebar.caption("Model: XGBoost")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Loyalty Program Overview")
    st.markdown("A high-level snapshot of the member base, churn risk, and segment distribution.")
    st.markdown("---")

    # ── KPI row ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Members",    f"{len(master):,}")
    col2.metric("Avg CLV (CAD)",    f"${master['CLV'].mean():,.0f}")
    col3.metric("Behavioral Churn", f"{master['churned_behavioral'].mean()*100:.1f}%")
    col4.metric("Formal Churn",     f"{master['churned_formal'].mean()*100:.1f}%")
    col5.metric("Critical Risk",    f"{(master['risk_tier']=='Critical').sum():,}")

    st.markdown("---")

    # ── Row 1: Segment distribution + Risk tier ──────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Member distribution by segment")
        seg_counts = master["segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [SEG_COLORS.get(s, "#999") for s in seg_counts.index]
        bars = ax.barh(seg_counts.index, seg_counts.values, color=colors, edgecolor="none")
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=10)
        ax.set_xlabel("Number of members")
        ax.set_xlim(0, seg_counts.max() * 1.15)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Risk tier distribution")
        risk_counts = master["risk_tier"].value_counts().reindex(["Low","Medium","High","Critical"])
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_r = [RISK_COLORS[r] for r in risk_counts.index]
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors_r, edgecolor="none", width=0.5)
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{val:,}", ha="center", fontsize=10)
        ax.set_ylabel("Number of members")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Row 2: CLV by segment + Loyalty card breakdown ───────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Average CLV by segment")
        clv_seg = master.groupby("segment")["CLV"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_c = [SEG_COLORS.get(s, "#999") for s in clv_seg.index]
        bars = ax.barh(clv_seg.index, clv_seg.values, color=colors_c, edgecolor="none")
        for bar, val in zip(bars, clv_seg.values):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                    f"${val:,.0f}", va="center", fontsize=9)
        ax.set_xlabel("Average CLV (CAD)")
        ax.set_xlim(0, clv_seg.max() * 1.18)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.subheader("Loyalty card distribution by segment")
        card_seg = pd.crosstab(master["segment"], master["Loyalty Card"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        card_seg.plot(kind="barh", stacked=True, ax=ax,
                      color=["#EF9F27","#7F77DD","#378ADD"], edgecolor="none")
        ax.set_xlabel("% of segment")
        ax.set_ylabel("")
        ax.legend(title="Card", bbox_to_anchor=(1,1), fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Province map table ───────────────────────────────────────────────
    st.subheader("Top provinces by member count and avg CLV")
    prov = master.groupby("Province").agg(
        members=("Loyalty Number","count"),
        avg_clv=("CLV","mean"),
        churn_rate=("churned_behavioral","mean"),
        critical_pct=("risk_tier", lambda x: (x=="Critical").mean()*100)
    ).round(2).sort_values("members", ascending=False).head(10)
    prov.columns = ["Members","Avg CLV (CAD)","Churn Rate","Critical Risk (%)"]
    st.dataframe(prov, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MEMBER LOOKUP
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Member Lookup":
    st.title("🔍 Member Lookup")
    st.markdown("Enter a Loyalty Number to see their risk score, segment, and recommended action.")
    st.markdown("---")

    loyalty_num = st.text_input("Enter Loyalty Number", placeholder="e.g. 100018")

    if loyalty_num:
        try:
            num = int(loyalty_num)
            member = master[master["Loyalty Number"] == num]

            if member.empty:
                st.error(f"No member found with Loyalty Number {num}")
            else:
                m = member.iloc[0]

                # ── Header ─────────────────────────────────────────────
                seg_color = SEG_COLORS.get(m["segment"], "#999")
                risk_color = RISK_COLORS.get(str(m["risk_tier"]), "#999")

                col1, col2, col3 = st.columns(3)
                col1.metric("Loyalty Number", f"{int(m['Loyalty Number'])}")
                col2.metric("Segment", m["segment"])
                col3.metric("Risk Tier", str(m["risk_tier"]))

                col4, col5, col6 = st.columns(3)
                col4.metric("Churn Probability", f"{m['churn_probability']*100:.2f}%")
                col5.metric("CLV (CAD)",          f"${m['CLV']:,.0f}")
                col6.metric("Loyalty Card",        m["Loyalty Card"])

                st.markdown("---")

                # ── Behavioral profile ──────────────────────────────────
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Behavioral profile")
                    profile_data = {
                        "Total flights (2yr)"    : int(m["total_flights"]),
                        "Active months / 24"     : int(m["months_with_flights"]),
                        "Activity rate"          : f"{m['activity_rate']*100:.1f}%",
                        "Recency (months ago)"   : int(m["recency_months"]),
                        "Redemption rate"        : f"{m['redemption_rate']*100:.1f}%",
                        "Activity trend"         : f"{m['activity_trend']:+.3f}",
                        "Avg flights last 6m"    : f"{m['avg_flights_last6m']:.2f}",
                        "Tenure (months)"        : int(m["tenure_months"]),
                    }
                    for k, v in profile_data.items():
                        st.write(f"**{k}:** {v}")

                with col_right:
                    st.subheader("Demographics")
                    demo_data = {
                        "Province"        : m["Province"],
                        "Gender"          : m["Gender"],
                        "Education"       : m["Education"],
                        "Marital Status"  : m["Marital Status"],
                        "Salary (CAD)"    : f"${m['Salary_imputed']:,.0f}",
                        "Enrollment Type" : m["Enrollment Type"],
                        "Enrollment Year" : int(m["Enrollment Year"]),
                    }
                    for k, v in demo_data.items():
                        st.write(f"**{k}:** {v}")

                st.markdown("---")

                # ── Recommended action ──────────────────────────────────
                st.subheader("🎯 Recommended retention action")

                playbook = {
                    "Loyal Frequent Fliers": {
                        "action"   : "Proactive reward — offer early access to tier upgrade 3 months before threshold. Send personalised milestone email with exclusive lounge access.",
                        "trigger"  : "Quarterly touchpoint — no specific trigger needed.",
                        "metric"   : "Tier upgrade conversion rate, NPS score",
                        "priority" : "🟢 Low urgency"
                    },
                    "At-Risk Inactives": {
                        "action"   : "Re-engagement email with time-limited double-points offer on next booking. If no response in 30 days, follow up with discounted companion fare.",
                        "trigger"  : "No flight activity for 6 consecutive months.",
                        "metric"   : "Reactivation rate within 90 days",
                        "priority" : "🔴 High urgency"
                    },
                    "Dormant High-Value": {
                        "action"   : "Personal outreach from account manager (not automated). Offer flexible booking credit worth 10% of historical annual spend.",
                        "trigger"  : "Activity trend negative AND no flight in last 4 months.",
                        "metric"   : "Return to at least 1 flight within 60 days",
                        "priority" : "🔴 High urgency"
                    },
                    "New Engagers": {
                        "action"   : "Onboarding nurture: Month 1 explain tier benefits, Month 2 highlight redemption options, Month 3 send first-anniversary bonus points.",
                        "trigger"  : "Enrolled less than 12 months ago.",
                        "metric"   : "90-day retention rate, first redemption rate",
                        "priority" : "🟡 Medium urgency"
                    },
                    "New Engagers II": {
                        "action"   : "Seasonal campaign targeting their historically active quarter. Offer bonus points on routes previously flown.",
                        "trigger"  : "Gap of more than 3 months between flights.",
                        "metric"   : "Booking rate from campaign, seasonal revenue per member",
                        "priority" : "🟡 Medium urgency"
                    },
                }

                seg = m["segment"]
                if seg in playbook:
                    p = playbook[seg]
                    st.info(f"**Priority:** {p['priority']}")
                    st.write(f"**Action:** {p['action']}")
                    st.write(f"**Trigger:** {p['trigger']}")
                    st.write(f"**Success metric:** {p['metric']}")
                else:
                    st.write("No specific playbook entry for this segment.")

        except ValueError:
            st.error("Please enter a valid numeric Loyalty Number.")

    else:
        # Show sample loyalty numbers to try
        st.info("💡 Try one of these sample Loyalty Numbers:")
        samples = master.groupby("segment").first()["Loyalty Number"].astype(int)
        for seg, num in samples.items():
            st.write(f"- **{seg}**: `{num}`")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEGMENT EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Segment Explorer":
    st.title("👥 Segment Explorer")
    st.markdown("Select a segment to explore its behavioral profile, risk distribution, and retention playbook.")
    st.markdown("---")

    segments = sorted(master["segment"].unique())
    selected = st.selectbox("Select a segment", segments)

    seg_data = master[master["segment"] == selected]
    seg_color = SEG_COLORS.get(selected, "#378ADD")

    st.markdown(f"### {selected}  —  {len(seg_data):,} members")

    # ── KPIs ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Members",          f"{len(seg_data):,}")
    col2.metric("Avg CLV",          f"${seg_data['CLV'].mean():,.0f}")
    col3.metric("Avg Churn Prob",   f"{seg_data['churn_probability'].mean()*100:.2f}%")
    col4.metric("Behavioral Churn", f"{seg_data['churned_behavioral'].mean()*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk tier breakdown")
        risk_dist = seg_data["risk_tier"].value_counts().reindex(["Low","Medium","High","Critical"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_r = [RISK_COLORS[r] for r in risk_dist.index]
        bars = ax.bar(risk_dist.index, risk_dist.values, color=colors_r, edgecolor="none", width=0.5)
        for bar, val in zip(bars, risk_dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{int(val):,}", ha="center", fontsize=10)
        ax.set_ylabel("Members")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Loyalty card mix")
        card_dist = seg_data["Loyalty Card"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_c = ["#EF9F27","#7F77DD","#378ADD"][:len(card_dist)]
        ax.pie(card_dist.values, labels=card_dist.index, colors=colors_c,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Behavioral stats ────────────────────────────────────────────────
    st.subheader("Behavioral profile vs overall average")
    compare_cols = ["activity_rate","recency_months","redemption_rate",
                    "avg_flights_last6m","total_flights","tenure_months"]
    compare_labels = ["Activity rate","Recency (months)","Redemption rate",
                      "Avg flights last 6m","Total flights","Tenure (months)"]

    seg_avg  = seg_data[compare_cols].mean()
    all_avg  = master[compare_cols].mean()

    compare_df = pd.DataFrame({
        "Feature"        : compare_labels,
        f"{selected}"    : seg_avg.values.round(2),
        "Overall avg"    : all_avg.values.round(2),
    })
    st.dataframe(compare_df.set_index("Feature"), use_container_width=True)

    st.markdown("---")

    # ── Retention playbook ──────────────────────────────────────────────
    st.subheader("🎯 Retention playbook for this segment")

    playbook = {
        "Loyal Frequent Fliers": {
            "priority" : "🟢 Low urgency — retain and reward",
            "who"      : "High activity rate, recent flights, very low churn risk. The airline's most engaged members.",
            "trigger"  : "Quarterly proactive touchpoint — no specific distress signal needed.",
            "action"   : "Offer early access to tier upgrade 3 months before they hit the threshold. Send a personalised milestone email (e.g. 500th flight) with exclusive lounge access or priority boarding.",
            "timeline" : "Quarterly. Milestone emails triggered automatically by flight count.",
            "metric"   : "Tier upgrade conversion rate, NPS score, year-on-year flight frequency.",
            "tradeoff" : "Low cost to serve. Risk of over-rewarding members who would stay anyway — monitor ROI per touchpoint."
        },
        "At-Risk Inactives": {
            "priority" : "🔴 High urgency — re-engage before permanent loss",
            "who"      : "Zero or near-zero flights in last 12+ months. Still enrolled but effectively gone.",
            "trigger"  : "No flight activity for 6 consecutive months.",
            "action"   : "Month 1: re-engagement email with time-limited double-points on next booking. Month 2 (if no booking): discounted companion fare on a popular route. Month 3: final win-back offer — if still no response, flag for programme removal.",
            "timeline" : "3-month escalating sequence from trigger date.",
            "metric"   : "Reactivation rate within 90 days. Revenue per reactivated member vs cost of campaign.",
            "tradeoff" : "High cost if companion fare is offered broadly. Limit Month 2 offer to members with CLV above $7,000."
        },
        "Dormant High-Value": {
            "priority" : "🔴 High urgency — highest revenue at risk",
            "who"      : "Above-average CLV but declining recent activity. These members are worth saving.",
            "trigger"  : "Activity trend negative AND no flight in last 4 months.",
            "action"   : "Personal outreach from account manager — not an automated email. Offer flexible booking credit worth 10% of their historical annual spend on the airline.",
            "timeline" : "Outreach within 2 weeks of trigger. Monthly check-in for 3 months.",
            "metric"   : "Return to at least 1 flight within 60 days. CLV retention rate.",
            "tradeoff" : "Account manager outreach does not scale. Limit this intervention to the top 500 Dormant High-Value members by CLV."
        },
        "New Engagers": {
            "priority" : "🟡 Medium urgency — build habits early",
            "who"      : "Low tenure (under 12 months), moderate activity. Still forming their loyalty habits.",
            "trigger"  : "Enrolled less than 12 months ago.",
            "action"   : "Automated 3-month onboarding sequence: Month 1 explain tier benefits and how points accumulate, Month 2 highlight redemption options with a worked example, Month 3 send first-anniversary bonus points (500 pts).",
            "timeline" : "Automated from enrollment date.",
            "metric"   : "90-day retention rate. First redemption rate within 6 months.",
            "tradeoff" : "Low marginal cost — fully automated. Risk of email fatigue if sequence is too frequent."
        },
        "New Engagers II": {
            "priority" : "🟡 Medium urgency — seasonal re-activation",
            "who"      : "Moderate tenure, regular but infrequent fliers. Fly in predictable seasonal patterns.",
            "trigger"  : "Gap of more than 3 months between flights.",
            "action"   : "Seasonal campaign sent 6 weeks before their historically active quarter. Offer bonus points (1.5x) on routes they have previously flown.",
            "timeline" : "Automated, triggered 6 weeks before each member's peak quarter.",
            "metric"   : "Booking rate from campaign. Seasonal revenue per member vs control group.",
            "tradeoff" : "Route-specific offers require data pipeline work. Start with top 10 most common routes to simplify."
        },
    }

    if selected in playbook:
        p = playbook[selected]
        st.warning(f"**{p['priority']}**")
        st.write(f"**Who they are:** {p['who']}")
        st.write(f"**Trigger:** {p['trigger']}")
        st.write(f"**Action:** {p['action']}")
        st.write(f"**Timeline:** {p['timeline']}")
        st.write(f"**Success metric:** {p['metric']}")
        st.write(f"**Trade-off to be aware of:** {p['tradeoff']}")
    else:
        st.write("No playbook entry for this segment.")
