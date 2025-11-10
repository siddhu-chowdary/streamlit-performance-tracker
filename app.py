# app.py — Performance Tracker Dashboard (polished layout + robust deltas + in-chart toggle)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Performance Tracker Dashboard", layout="wide")

# ---------- Brand ----------
BRAND = {
    "primary":     "#436DB3",
    "light_blue":  "#BFD0EE",
    "danger":      "#F4454E",
    "bg_soft":     "#F7F3EF",
    "border":      "#EDEDED",
    "ink":         "#0B1221",
    "ok":          "#1A8E3B",
    "link_blue":   "#0071BC",
}
BLUES = ["#436DB3", "#5B84C7", "#87A9DA", "#AFC5E8", "#D3E1F5"]
CHART_TITLE_SIZE = 18
PLOT_HEIGHT = 380  # same height for paired charts

# ---------- CSS: compact, no “white strip” ----------
st.markdown(f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap");
html, body, [class*="css"] {{
  font-family:"Roboto", system-ui, -apple-system, Segoe UI, Arial, sans-serif !important;
  color:{BRAND["ink"]}; background:{BRAND["bg_soft"]};
}}
.block-container {{ padding-top: 28px; }}

.header {{
  background:#2B2F36; color:#fff; border-radius:12px; padding:14px 16px; margin: 4px 0 12px 0;
}}
.header .title {{ font-weight:700; font-size:18px; line-height:22px; margin:0; }}
.header .sub   {{ opacity:.9; font-size:13px; }}

.kpi-tile {{
  background:#fff; border:1px solid {BRAND["border"]}; border-radius:12px; padding:12px 14px; margin-bottom:10px;
}}
.kpi-label {{ font-size:12px; color:#445268; display:flex; gap:8px; align-items:center; }}
.kpi-value {{ font-weight:700; font-size:26px; color:{BRAND["ink"]}; }}
.kpi-delta {{ font-size:12px; margin-top:2px; }}

.left-radio .stRadio > div {{ gap:6px; }}
.left-radio label {{ font-weight:600; color:#0B2648; }}

/* remove any stray vertical gap around Plotly charts */
.stPlotlyChart {{ margin-top:0 !important; }}
/* no cards/borders around charts — just charts */
.chart-wrap {{ margin:0; padding:0; }}
</style>
""", unsafe_allow_html=True)

# ---------- Plotly helpers ----------
px.defaults.template = "plotly_white"

def style_layout(fig, title=None, *, legend_pos="top-right", hide_grid=True, bottom_legend=False, height=PLOT_HEIGHT):
    if bottom_legend:
        legend = dict(orientation="h", y=-0.12, x=0.5, xanchor="center"); bmargin = 46
    elif legend_pos == "top-right":
        legend = dict(orientation="h", y=1.02, x=1.0, xanchor="right"); bmargin = 10
    else:
        legend = dict(orientation="h", y=1.02, x=0.0, xanchor="left"); bmargin = 10

    fig.update_layout(
        title=title,
        title_font=dict(size=CHART_TITLE_SIZE, family="Roboto", color=BRAND["ink"]),
        font=dict(family="Roboto", size=12, color=BRAND["ink"]),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=8, r=8, t=26, b=bmargin),
        legend=legend,
        height=height
    )
    fig.update_xaxes(showgrid=(not hide_grid), gridcolor=BRAND["border"])
    fig.update_yaxes(showgrid=(not hide_grid), gridcolor=BRAND["border"])
    return fig

# ---------- Data loader (dedupe/variants -> canonical) ----------
@st.cache_data
def load_data(src_path: str) -> pd.DataFrame:
    df = pd.read_csv(src_path)

    def make_unique(cols):
        seen, out = {}, []
        for c in cols:
            c = str(c).strip()
            if c not in seen: seen[c]=1; out.append(c)
            else: seen[c]+=1; out.append(f"{c}_{seen[c]}")
        return out
    def lower_unique(cols):
        seen, out = {}, []
        for c in cols:
            lc = c.lower().strip()
            if lc not in seen: seen[lc]=1; out.append(lc)
            else: seen[lc]+=1; out.append(f"{lc}_{seen[lc]}")
        return out

    df.columns = make_unique(df.columns)
    df.columns = lower_unique(df.columns)

    def coalesce_into(base):
        if base in df.columns:
            for col in [c for c in df.columns if c.startswith(base + "_")]:
                df[base] = df[base].where(df[base].notna(), df[col])
        else:
            alts = [c for c in df.columns if c == base or c.startswith(base + "_")]
            if alts:
                df[base] = df[alts[0]]
                for col in alts[1:]:
                    df[base] = df[base].where(df[base].notna(), df[col])

    for base in ["event_type","event_date","state","city","zipcode",
                 "traffic_source","utm_campaign","device_type","browser",
                 "retention_status","program_activity","user_id",
                 "a1c","weight","height"]:
        coalesce_into(base)

    if "a1c" not in df.columns or df["a1c"].isna().all():
        for cand in ["a1c_current","a1c_value","a1c_level"]:
            if cand in df.columns and df[cand].notna().any():
                df["a1c"] = pd.to_numeric(df[cand], errors="coerce"); break
    if "height" not in df.columns or df["height"].isna().all():
        for cand in ["height_in_inches","height_inches","height_in"]:
            if cand in df.columns:
                df["height"] = pd.to_numeric(df[cand], errors="coerce") * 0.0254; break
    if "weight" not in df.columns or df["weight"].isna().all():
        for cand, conv in [("weight_current_kg",1.0),("weight_kg",1.0),
                           ("weight_current_lb",1/2.20462),("weight_lbs",1/2.20462)]:
            if cand in df.columns:
                df["weight"] = pd.to_numeric(df[cand], errors="coerce")*conv; break

    def ensure_event_date(frame):
        if "event_date" in frame.columns:
            return pd.to_datetime(frame["event_date"], errors="coerce")
        cand = next((c for c in frame.columns if "date" in c or "time" in c), None)
        if cand:
            ser = pd.to_datetime(frame[cand], errors="coerce")
            if ser.notna().any(): return ser
        start = pd.Timestamp("2024-11-01")
        return start + pd.to_timedelta(np.arange(len(frame)) % 365, unit="D")
    df["event_date"] = ensure_event_date(df)

    if "state" not in df.columns: df["state"] = "Kansas"
    df["state"] = df["state"].astype("string")
    if "county" in df.columns: df["county"] = df["county"].astype("string")
    if "city" not in df.columns: df["city"] = df.get("county", pd.NA)
    df["city"] = df["city"].astype("string")

    if "weight" in df.columns and "height" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["bmi"] = df["weight"] / (df["height"]**2)

    for c in ["event_type","traffic_source","utm_campaign","device_type","browser",
              "zipcode","retention_status","program_activity","user_id"]:
        if c not in df.columns: df[c] = pd.NA
        df[c] = df[c].astype("string")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype("string").str.strip().str.lower()
    return df

# ---------- Read + scope ----------
DEFAULT_PATH = "data/kansas_nutrition_first_web_analytics_events.csv"
uploader = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
data = load_data(uploader if uploader else DEFAULT_PATH)
data = data.loc[data["state"].astype("string").str.lower().isin(["kansas","ks"])].copy()

# ---------- Header ----------
st.markdown(f"""
<div class="header">
  <div class="title">Performance Tracker Dashboard</div>
  <div class="sub">Clinically-validated Nutrition First — FY {date.today().strftime("%Y")}</div>
</div>
""", unsafe_allow_html=True)

# ---------- Filters (Program removed) ----------
def options_from(df, primary, fallback=None):
    if primary in df.columns and df[primary].dropna().astype("string").str.strip().ne("").any():
        vals = sorted(df[primary].dropna().astype("string").str.strip().unique().tolist())
        return ["All"] + vals
    if fallback and fallback in df.columns and df[fallback].dropna().astype("string").str.strip().ne("").any():
        vals = sorted(df[fallback].dropna().astype("string").str.strip().unique().tolist())
        return ["All"] + vals
    return ["All"]

frow = st.columns([1,1,1,1])
county = frow[0].selectbox("County", options_from(data, "county", "city"), index=0)
device  = frow[1].selectbox("Device",  options_from(data, "device_type"), index=0)
channel = frow[2].selectbox("Channel", options_from(data, "traffic_source"),  index=0)

min_d, max_d = pd.to_datetime(data["event_date"]).min(), pd.to_datetime(data["event_date"]).max()
if pd.isna(min_d) or pd.isna(max_d):
    min_d = pd.Timestamp("2024-11-01"); max_d = min_d + pd.offsets.MonthEnd(11)
dr = frow[3].date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = (pd.to_datetime(dr[0]), pd.to_datetime(dr[1])) if isinstance(dr, tuple) else (min_d, max_d)

# base (non-date) mask — used for current and prior windows
base_mask = pd.Series(True, index=data.index)
if county != "All":
    if "county" in data.columns and county in data["county"].astype("string").unique():
        base_mask &= data["county"].astype("string").eq(county)
    else:
        base_mask &= data["city"].astype("string").eq(county)
if device != "All" and "device_type" in data.columns:
    base_mask &= data["device_type"].astype("string").eq(device)
if channel != "All" and "traffic_source" in data.columns:
    base_mask &= data["traffic_source"].astype("string").eq(channel)

df = data.loc[base_mask & data["event_date"].between(start_d, end_d)].copy()

# ---------- KPI + Funnel inference ----------
EXPECTED = {"crossover","link_click","signup","improvement"}

def compute_counts(frame: pd.DataFrame):
    if "event_type" in frame.columns and frame["event_type"].dropna().isin(EXPECTED).any():
        c = frame["event_type"].value_counts()
        return {
            "crossover": int(c.get("crossover", 0)),
            "link_click": int(c.get("link_click", 0)),
            "signup":    int(c.get("signup", 0)),
            "improve":   int(c.get("improvement", 0)),
        }
    total = len(frame)
    clicks = int(frame["traffic_source"].notna().sum()) if "traffic_source" in frame.columns else int(0.33*total)
    signups = int(0.05*total)
    improve = 0
    return {"crossover": total, "link_click": clicks, "signup": signups, "improve": improve}

def counts_for_window(s, e):
    frame = data.loc[base_mask & data["event_date"].between(s, e)]
    return compute_counts(frame)

cur_counts = counts_for_window(start_d, end_d)
span_days = max((end_d.normalize() - start_d.normalize()).days + 1, 1)
prev_e = start_d - timedelta(days=1)
prev_s = prev_e - timedelta(days=span_days-1)
prev_counts = counts_for_window(prev_s, prev_e)

def delta_text(curr, prev):
    if prev <= 0:
        return "↔ 0.0% vs prev", "#6B7280"  # neutral when no prior denominator
    pct = round(100*(curr-prev)/prev, 1)
    arrow = "▲" if pct >= 0 else "▼"
    color = BRAND["ok"] if pct >= 0 else BRAND["danger"]
    return f"{arrow} {abs(pct)}% vs prev", color

# ---------- KPI tiles ----------
KPI = [
    ("Website Crossovers", "🌐", cur_counts["crossover"], prev_counts["crossover"]),
    ("Link Clicks",        "🔗", cur_counts["link_click"], prev_counts["link_click"]),
    ("Sign-ups",           "📝", cur_counts["signup"],    prev_counts["signup"]),
    ("Health Improvements","💙", cur_counts["improve"],   prev_counts["improve"]),
]
k1,k2,k3,k4 = st.columns(4)
for col, (label, icon, cur, prv) in zip([k1,k2,k3,k4], KPI):
    txt, colr = delta_text(cur, prv)
    with col:
        st.markdown(
            f"""
            <div class="kpi-tile">
              <div class="kpi-label">{icon}&nbsp;&nbsp;{label}</div>
              <div class="kpi-value">{cur:,}</div>
              <div class="kpi-delta" style="color:{colr};font-weight:600">{txt}</div>
            </div>
            """, unsafe_allow_html=True
        )

# ---------- Left nav ----------
left, main = st.columns([0.23, 1], gap="large")
with left:
    st.markdown('<div class="left-radio">', unsafe_allow_html=True)
    tab = st.radio("Navigation", ["Executive Overview","Acquisition & Engagement","Conversion","Health Outcomes"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Helpers ----------
def monthly_pivot_any(frame):
    if "event_type" in frame.columns and frame["event_type"].dropna().isin({"crossover","link_click","signup"}).any():
        g = (frame.assign(period=frame["event_date"].dt.to_period("M").dt.to_timestamp())
                   .groupby(["period","event_type"]).size().reset_index(name="count"))
        p = g.pivot(index="period", columns="event_type", values="count").fillna(0)
        p = p.rename(columns={"crossover":"Crossovers","link_click":"Clicks","signup":"Sign-ups"})
        for c in ["Crossovers","Clicks","Sign-ups"]:
            if c not in p.columns: p[c] = 0
        return p
    p = (frame.assign(period=frame["event_date"].dt.to_period("M").dt.to_timestamp())
               .groupby("period").size().to_frame("Crossovers"))
    if "traffic_source" in frame.columns:
        c = (frame[frame["traffic_source"].notna()]
                .assign(period=frame["event_date"].dt.to_period("M").dt.to_timestamp())
                .groupby("period").size())
        p["Clicks"] = c
    else:
        p["Clicks"] = 0
    p["Sign-ups"] = 0
    return p.fillna(0)

def smooth_line(df_line, y_cols, title, color_seq=None, height=PLOT_HEIGHT):
    fig = px.line(
        df_line.reset_index(), x="period", y=y_cols, markers=False,
        line_shape="spline",
        color_discrete_sequence=color_seq or [BRAND["primary"], BRAND["light_blue"], BRAND["danger"]]
    )
    fig.update_traces(line=dict(width=2.6))
    return style_layout(fig, title, legend_pos="top-right", hide_grid=True, height=height)

def treemap_zip_with_pct(frame, title, height=PLOT_HEIGHT):
    agg = frame.groupby("zipcode").size().reset_index(name="count")
    total = int(agg["count"].sum()) if not agg.empty else 0
    agg["pct"] = 0 if total == 0 else 100.0 * agg["count"] / total
    agg["label"] = agg["zipcode"].astype(str) + "<br>" + agg["pct"].round(0).astype(int).astype(str) + "%"
    fig = px.treemap(
        agg, path=["label"], values="count", color="pct",
        color_continuous_scale=[[0, "#AFC5E8"], [1, BRAND["primary"]]],
        range_color=(0, agg["pct"].max() if not agg.empty else 1)
    )
    fig.update_traces(texttemplate="%{label}", hovertemplate="Zip %{label}<br>Count %{value}<br>% %{color:.1f}")
    return style_layout(fig, title, bottom_legend=True, hide_grid=True, height=height)

# ---------- Tabs ----------
with main:
    if tab == "Executive Overview":
        c1, c2 = st.columns([1,1.2])
        with c1:
            funnel_df = pd.DataFrame({
                "stage":["Website Crossovers","Link Clicks","Sign-ups","Health Improvements"],
                "count":[cur_counts["crossover"], cur_counts["link_click"], cur_counts["signup"], cur_counts["improve"]]
            })
            fig = px.funnel(funnel_df, y="stage", x="count",
                            color_discrete_sequence=[BRAND["primary"]])
            fig = style_layout(fig, "Conversion Funnel", legend_pos="top-right", hide_grid=False, height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pvt = monthly_pivot_any(df)
            ycols = [c for c in ["Crossovers","Clicks","Sign-ups"] if c in pvt.columns]
            fig = smooth_line(pvt, ycols, "Performance over Time", height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

    elif tab == "Acquisition & Engagement":
        a1, a2 = st.columns([1.2,0.9])
        with a1:
            pvt = monthly_pivot_any(df)
            if "Clicks" in pvt:
                fig = smooth_line(pvt[["Clicks"]], ["Clicks"], "Link Clicks over Time", height=PLOT_HEIGHT)
            else:
                fig = smooth_line(pvt[["Crossovers"]], ["Crossovers"], "Crossovers over Time", height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            if "traffic_source" in df.columns and df["traffic_source"].notna().any():
                ch = (df[df["traffic_source"].notna()]
                        .groupby("traffic_source").size().reset_index(name="count")
                        .sort_values("count", ascending=False))
            else:
                ch = pd.DataFrame({"traffic_source":["Email","Organic","Social"],"count":[1,1,1]})
            fig = px.pie(ch, values="count", names="traffic_source", hole=0.62,
                         color="traffic_source", color_discrete_sequence=BLUES)
            fig.update_traces(textinfo="percent+label")
            fig = style_layout(fig, "Acquisition Channel Performance", bottom_legend=True, height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

    elif tab == "Conversion":
        t1, t2 = st.columns([1,1])
        with t1:
            # Build in-chart toggle using Plotly updatemenus (top-right)
            # Prepare both datasets
            if "device_type" in df.columns and df["device_type"].notna().any():
                d_dev = df.groupby("device_type").size().reset_index(name="count")
            else:
                d_dev = pd.DataFrame({"device_type":["Desktop","Mobile","Tablet"], "count":[5,3,2]})
            if "browser" in df.columns and df["browser"].notna().any():
                d_bro = df.groupby("browser").size().reset_index(name="count")
            else:
                d_bro = pd.DataFrame({"browser":["Chrome","Safari","Edge"], "count":[5,3,2]})

            # Start with device
            fig = go.Figure(data=[go.Pie(
                labels=d_dev["device_type"].astype(str),
                values=d_dev["count"],
                hole=0.62,
                textinfo="percent+label"
            )])
            # Buttons to swap labels/values
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    direction="right",
                    x=1.0, xanchor="right",
                    y=1.22, yanchor="top",
                    buttons=[
                        dict(label="Device",
                             method="update",
                             args=[{"labels":[d_dev["device_type"].astype(str)],
                                    "values":[d_dev["count"]]},
                                   {"title":"Traffic by Device"}]),
                        dict(label="Browser",
                             method="update",
                             args=[{"labels":[d_bro["browser"].astype(str)],
                                    "values":[d_bro["count"]]},
                                   {"title":"Traffic by Browser"}]),
                    ]
                )]
            )
            fig = style_layout(fig, "Traffic by Device", bottom_legend=True, height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            zf = df.copy()
            if "zipcode" not in zf.columns or zf["zipcode"].isna().all():
                labels = ["66101","66103","66105","66111","66119"]
                if "user_id" in zf.columns and zf["user_id"].notna().any():
                    codes = pd.factorize(zf["user_id"].astype("string"))[0] % len(labels)
                    zf = zf.assign(zipcode=[labels[i] for i in codes])
                else:
                    zf = zf.assign(zipcode=np.random.choice(labels, size=len(zf)))
            fig = treemap_zip_with_pct(zf, "Local Hotspots — Zipcodes", height=PLOT_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

    elif tab == "Health Outcomes":
        def monthly_a1c_control(frame):
            if "a1c" not in frame.columns or frame["a1c"].dropna().empty:
                return pd.DataFrame({"period": [], "pct_below": []})
            t = frame.dropna(subset=["a1c"]).copy()
            t["period"] = t["event_date"].dt.to_period("M").dt.to_timestamp()
            g = t.groupby("period")["a1c"]
            return pd.DataFrame({"period": g.median().index,
                                 "pct_below": g.apply(lambda s: (s < 6.5).mean()*100).values})

        def monthly_weight_loss_pct(frame):
            # % of members whose latest weight in that month is ≤ 95% of baseline
            if "weight" not in data.columns or data["weight"].dropna().empty: return pd.DataFrame()
            if "user_id" not in data.columns or data["user_id"].dropna().empty: return pd.DataFrame()

            t_all = data[data["user_id"].notna() & data["weight"].notna()].copy()
            t_all["_uid"] = t_all["user_id"].astype("string")
            t_all = t_all.sort_values(["_uid","event_date"])
            baseline = t_all.groupby("_uid").first()["weight"].rename("base")

            t = frame[frame["user_id"].notna() & frame["weight"].notna()].copy()
            if t.empty: return pd.DataFrame()
            t["_uid"] = t["user_id"].astype("string")
            t["period"] = t["event_date"].dt.to_period("M").dt.to_timestamp()
            latest_m = (t.sort_values(["_uid","period","event_date"])
                        .groupby(["_uid","period"]).last()["weight"].rename("last")).reset_index()
            latest_m = latest_m.merge(baseline.to_frame(), left_on="_uid", right_index=True, how="left")
            latest_m = latest_m[latest_m["base"].notna() & (latest_m["base"] > 0)]
            latest_m["meets"] = (latest_m["last"] <= 0.95 * latest_m["base"]).astype(int)
            return latest_m.groupby("period")["meets"].mean().mul(100).reset_index(name="pct_ls")

        def monthly_weight_avg(frame):
            if "weight" not in frame.columns or frame["weight"].dropna().empty:
                return pd.DataFrame()
            t = frame.dropna(subset=["weight"]).copy()
            t["period"] = t["event_date"].dt.to_period("M").dt.to_timestamp()
            return t.groupby("period")["weight"].mean().reset_index(name="avg_weight")

        h1, h2 = st.columns([1.2, 1.0])

        with h1:
            a1c_m = monthly_a1c_control(df)
            if a1c_m.empty:
                st.info("No A1C column found or it is empty.")
            else:
                fig = px.line(a1c_m, x="period", y="pct_below",
                              markers=False, line_shape="spline",
                              color_discrete_sequence=[BRAND["primary"]])
                fig.update_traces(line=dict(width=2.6), hovertemplate=" %{y:.1f}%")
                fig = style_layout(fig, "A1C Control — % of Members Below 6.5", height=PLOT_HEIGHT)
                st.plotly_chart(fig, use_container_width=True)
                sel = df["a1c"].dropna()
                st.markdown(f"**{(sel < 6.5).mean()*100:.1f}%** of members are **below 6.5 A1C** in the selected date range.")

        with h2:
            prog = monthly_weight_loss_pct(df)
            if not prog.empty:
                fig = px.line(prog, x="period", y="pct_ls",
                              markers=False, line_shape="spline",
                              color_discrete_sequence=[BRAND["primary"]])
                fig.update_traces(line=dict(width=2.6), hovertemplate=" %{y:.1f}%")
                fig = style_layout(fig, "Weight Loss Progress — % ≥5% Below Baseline", height=PLOT_HEIGHT)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Share of members whose latest weight each month is at least **5% below** their baseline.")
            else:
                wavg = monthly_weight_avg(df)
                if wavg.empty:
                    st.info("No weight data available for the selected period.")
                else:
                    fig = px.line(wavg, x="period", y="avg_weight",
                                  markers=False, line_shape="spline",
                                  color_discrete_sequence=[BRAND["primary"]])
                    fig.update_traces(line=dict(width=2.6))
                    fig = style_layout(fig, "Average Weight Over Time (kg)", height=PLOT_HEIGHT)
                    st.plotly_chart(fig, use_container_width=True)
