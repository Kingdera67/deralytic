import io
import os
import re
from datetime import date
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import URL
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Deralytic", page_icon="\U0001F4CA", layout="wide")
AUTHOR = "Shalom Chidera Reuben"  # change this to your name if you like

st.title("\U0001F4CA Deralytic")
st.caption("Clean, explore and report on your data. Upload a file or connect to a database to begin.")
with st.expander("How to use Deralytic", expanded=False):
    st.markdown(
        """
1. **Load data:** upload a CSV or Excel file, connect to a database, or try the sample data.
2. **Clean it:** choose automatic cleaning in the sidebar, or fix one column at a time in *Clean columns*.
3. **Explore:** see statistics, charts and plain-language insights.
4. **Predict:** train a model to predict a column, or group similar rows into segments.
5. **Report:** download a PDF summary, or save the cleaned data as CSV, Excel or a database table.
"""
    )


def footer():
    st.divider()
    st.caption(f"Deralytic \u00b7 built by {AUTHOR} with Python, pandas, scikit-learn and Streamlit")


# ---------- helpers ----------
def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin1"):
            try:
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding=enc)
            except UnicodeDecodeError:
                continue
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")


ID_WORDS = {"id", "code", "zip", "postal", "phone", "number", "sku"}


def looks_like_id(col):
    tokens = set(re.split(r"[^a-z0-9]+", str(col).lower()))
    return bool(tokens & ID_WORDS)


def text_cols(df):
    return [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def clean(df, opts):
    df = df.copy()
    log = []

    if opts["names"]:
        df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
        log.append("Standardised column names")

    if opts["strip"]:
        for c in text_cols(df):
            df[c] = df[c].str.strip().replace("", np.nan)
        log.append("Trimmed extra spaces in text columns")

    if opts["types"]:
        for c in text_cols(df):
            as_num = pd.to_numeric(df[c], errors="coerce")
            filled = df[c].notna().sum()
            if not looks_like_id(c) and filled > 0 and as_num.notna().sum() / filled > 0.9:
                df[c] = as_num
                log.append(f"Converted '{c}' to numbers")
                continue
            if any(k in c.lower() for k in ("date", "time")):
                as_dt = pd.to_datetime(df[c], errors="coerce")
                if filled > 0 and as_dt.notna().sum() / filled > 0.8:
                    df[c] = as_dt
                    log.append(f"Converted '{c}' to dates")

    if opts["dupes"]:
        before = len(df)
        df = df.drop_duplicates()
        log.append(f"Removed {before - len(df)} duplicate rows")

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = text_cols(df)

    if opts["num_fill"] == "Drop rows":
        before = len(df)
        df = df.dropna(subset=num_cols)
        log.append(f"Dropped {before - len(df)} rows with missing numbers")
    elif opts["num_fill"] != "Leave as is":
        for c in num_cols:
            if df[c].isna().any():
                val = {"Median": df[c].median(), "Mean": df[c].mean(), "Zero": 0}[opts["num_fill"]]
                df[c] = df[c].fillna(val)
        log.append(f"Filled missing numbers with {opts['num_fill'].lower()}")

    if opts["cat_fill"] != "Leave as is":
        for c in cat_cols:
            if df[c].isna().any():
                if opts["cat_fill"] == "Most common value" and not df[c].mode().empty:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
                else:
                    df[c] = df[c].fillna("Unknown")
        log.append(f"Filled missing text with {opts['cat_fill'].lower()}")

    if opts["outliers"]:
        for c in df.select_dtypes(include="number").columns:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        log.append("Capped outliers using the IQR rule")

    return df, log


def make_insights(df):
    out = [f"The dataset has {len(df):,} rows and {df.shape[1]} columns."]
    miss = df.isna().mean()
    for c, v in miss[miss > 0.3].items():
        out.append(f"'{c}' is {v:.0%} missing. Consider dropping it or collecting more data.")
    num = df.select_dtypes(include="number")
    if num.shape[1] >= 2:
        corr = num.corr().abs()
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if corr.loc[a, b] > 0.7:
                    out.append(f"'{a}' and '{b}' are strongly related (correlation {num[a].corr(num[b]):.2f}).")
    for c in num.columns:
        if abs(num[c].skew()) > 1:
            out.append(f"'{c}' is skewed (skew {num[c].skew():.1f}); the median may describe it better than the mean.")
    for c in df.select_dtypes(exclude=["number", "datetime"]).columns:
        if 1 < df[c].nunique() <= 30:
            share = df[c].value_counts(normalize=True)
            out.append(f"In '{c}', '{share.index[0]}' is the most common value ({share.iloc[0]:.0%}).")
    return out



# ---------- column-by-column steps (stage 2) ----------
def describe_step(st_):
    t, c = st_["type"], st_.get("col")
    if t == "drop":
        return f"Dropped column '{c}'"
    if t == "rename":
        return f"Renamed '{c}' to '{st_['new']}'"
    if t == "to_type":
        return f"Changed '{c}' to {st_['to'].lower()}"
    if t == "fill":
        extra = f" ({st_['value']})" if st_["method"] == "Custom value" else ""
        return f"Filled missing values in '{c}' with {st_['method'].lower()}{extra}"
    if t == "replace":
        return f"In '{c}', replaced '{st_['old']}' with '{st_['new']}'"
    if t == "drop_missing":
        return f"Dropped rows where '{c}' is missing"
    return str(st_)


def apply_step(df, st_):
    t, c = st_["type"], st_.get("col")
    if c not in df.columns:
        raise KeyError(f"column '{c}' not found")
    df = df.copy()
    is_num = pd.api.types.is_numeric_dtype(df[c])

    if t == "drop":
        df = df.drop(columns=[c])
    elif t == "rename":
        df = df.rename(columns={c: st_["new"]})
    elif t == "to_type":
        if st_["to"] == "Number":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif st_["to"] == "Date":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v)).astype("object")
    elif t == "fill":
        m = st_["method"]
        if m == "Median":
            val = df[c].median()
        elif m == "Mean":
            val = df[c].mean()
        elif m == "Zero":
            val = 0
        elif m == "Most common value":
            val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
        else:
            val = pd.to_numeric(st_["value"], errors="coerce") if is_num else st_["value"]
        df[c] = df[c].fillna(val)
    elif t == "replace":
        new = pd.to_numeric(st_["new"], errors="coerce") if is_num else st_["new"]
        mask = df[c].astype(str) == st_["old"]
        df.loc[mask, c] = new
    elif t == "drop_missing":
        df = df.dropna(subset=[c])
    return df


def apply_steps(df, steps):
    warnings = []
    for i, st_ in enumerate(steps, 1):
        try:
            df = apply_step(df, st_)
        except Exception as e:
            warnings.append(f"Step {i} skipped: {e}")
    return df, warnings



# ---------- one-click PDF report (stage 3) ----------
BLUE = colors.HexColor("#1f5f8b")


def short(x, n=26):
    x = str(x)
    return x if len(x) <= n else x[: n - 1] + "..."


def make_table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def fig_to_image(fig, width=6.3 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = PILImage.open(io.BytesIO(buf.getvalue())).size
    buf.seek(0)
    return Image(buf, width=width, height=width * h / w)


def build_report(title, raw, df, log, insights, charts=True):
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, textColor=BLUE)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=BLUE, spaceBefore=10)
    body = styles["BodyText"]

    story = [
        Paragraph("DERALYTIC", ParagraphStyle("brand", parent=styles["Normal"], fontSize=9, textColor=BLUE, alignment=1)),
        Paragraph(escape(title), h1),
        Paragraph(f"Generated by Deralytic on {date.today():%d %B %Y}", body),
        Spacer(1, 6),
        Paragraph("Overview", h2),
        make_table([
            ["Rows (raw)", f"{len(raw):,}"],
            ["Rows (cleaned)", f"{len(df):,}"],
            ["Columns (cleaned)", str(df.shape[1])],
            ["Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}"],
            ["Missing cells (cleaned)", f"{int(df.isna().sum().sum()):,}"],
        ], col_widths=[2.2 * inch, 1.5 * inch], header=False),
    ]

    story.append(Paragraph("What was cleaned", h2))
    for item in log or ["No cleaning steps were applied."]:
        story.append(Paragraph("\u2022 " + escape(item), body))

    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False).head(10)
    if len(miss):
        story.append(Paragraph("Missing values in the raw data", h2))
        rows = [["Column", "Missing", "Percent"]] + [
            [short(c), f"{int(v):,}", f"{v / len(raw) * 100:.1f}%"] for c, v in miss.items()
        ]
        story.append(make_table(rows, col_widths=[2.8 * inch, 1.1 * inch, 1.1 * inch]))

    num = df.select_dtypes(include="number")
    if num.shape[1]:
        story.append(Paragraph("Summary statistics", h2))
        desc = num.iloc[:, :8].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
        rows = [["Column", "Mean", "Std", "Min", "Median", "Max"]] + [
            [short(c)] + [f"{v:,.2f}" for v in r] for c, r in desc.iterrows()
        ]
        story.append(make_table(rows, col_widths=[1.9 * inch] + [0.9 * inch] * 5))

    story.append(Paragraph("Key insights", h2))
    for line in insights:
        story.append(Paragraph("\u2022 " + escape(line), body))

    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", h2))
        if num.shape[1] >= 2:
            cols = list(num.columns[:10])
            corr = num[cols].corr()
            fig, ax = plt.subplots(figsize=(6.5, 5))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)), [short(c, 14) for c in cols], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(cols)), [short(c, 14) for c in cols], fontsize=7)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Correlation between numeric columns")
            story += [fig_to_image(fig), Spacer(1, 8)]
        for c in df.select_dtypes(exclude=["number", "datetime"]).columns:
            if 2 <= df[c].nunique() <= 30:
                vc = df[c].value_counts().head(10).sort_values()
                fig, ax = plt.subplots(figsize=(6.5, 3.5))
                ax.barh([short(i, 24) for i in vc.index], vc.values, color="#2e86ab")
                ax.set_title(f"Most common values in '{short(c, 30)}'")
                story += [fig_to_image(fig), Spacer(1, 8)]
                break
        if num.shape[1]:
            c = num.columns[0]
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.hist(num[c].dropna(), bins=30, color="#2e86ab")
            ax.set_title(f"Distribution of '{short(c, 30)}'")
            story.append(fig_to_image(fig))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54,
                      bottomMargin=54, title=title).build(story)
    return buf.getvalue()



# ---------- machine learning (stage 4) ----------
def candidate_features(df, target):
    out = []
    for c in df.columns:
        if c == target or looks_like_id(c):
            continue
        sr = df[c]
        if pd.api.types.is_datetime64_any_dtype(sr) or pd.api.types.is_numeric_dtype(sr):
            out.append(c)
        elif sr.nunique() <= 50:
            out.append(c)
    return out


def detect_task(y):
    return "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 else "classification"


def expand_dates(X):
    X = X.copy()
    for c in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[f"{c}_year"] = X[c].dt.year
            X[f"{c}_month"] = X[c].dt.month
            X[f"{c}_dayofweek"] = X[c].dt.dayofweek
            X = X.drop(columns=[c])
    return X


def train_model(df, target, features, test_size=0.25):
    data = df.dropna(subset=[target]).copy()
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    y = data[target]
    X = expand_dates(data[features])
    task = detect_task(y)
    num = list(X.select_dtypes(include="number").columns)
    cat = [c for c in X.columns if c not in num]
    for c in cat:
        X[c] = X[c].astype(str)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    if task == "regression":
        model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    res = {"task": task, "target": target, "n_train": len(X_tr), "n_test": len(X_te),
           "y_test": y_te.reset_index(drop=True), "pred": pd.Series(pred)}
    if task == "regression":
        res["r2"] = r2_score(y_te, pred)
        res["mae"] = mean_absolute_error(y_te, pred)
        res["rmse"] = float(np.sqrt(mean_squared_error(y_te, pred)))
        res["baseline_mae"] = mean_absolute_error(y_te, np.full(len(y_te), y_tr.median()))
    else:
        res["accuracy"] = accuracy_score(y_te, pred)
        res["f1"] = f1_score(y_te, pred, average="weighted")
        res["baseline"] = float((y_te == y_tr.mode().iloc[0]).mean())
        labels = sorted(y.astype(str).unique()) if y.dtype == object else sorted(y.unique())
        res["labels"] = [str(x) for x in labels]
        res["cm"] = confusion_matrix(y_te, pred, labels=labels)

    n = min(len(X_te), 2000)
    r = permutation_importance(pipe, X_te.iloc[:n], y_te.iloc[:n], n_repeats=3, random_state=42, n_jobs=1)
    res["importance"] = (
        pd.DataFrame({"column": X_te.columns, "importance": r.importances_mean})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    return res


def explain_result(res):
    t = res["target"]
    lines = []
    if res["task"] == "regression":
        lines.append(
            f"On rows it had not seen, the model explains {max(res['r2'], 0):.0%} of the variation in '{t}'. "
            f"Its predictions are off by about {res['mae']:,.2f} on average, compared with {res['baseline_mae']:,.2f} "
            "if you simply guessed the typical value."
        )
        if res["r2"] < 0.2:
            lines.append("This is a weak result. The chosen columns do not explain much about this target.")
        if res["r2"] > 0.98:
            lines.append("The score is suspiciously high. Check for a column that gives the answer away, such as a copy of the target.")
    else:
        lines.append(
            f"The model predicted '{t}' correctly {res['accuracy']:.0%} of the time on rows it had not seen. "
            f"Always guessing the most common value would be right {res['baseline']:.0%} of the time."
        )
        if res["accuracy"] <= res["baseline"] + 0.02:
            lines.append("The model is no better than guessing the most common value, so these columns do not predict this target well.")
        if res["accuracy"] > 0.98 and res["accuracy"] > res["baseline"] + 0.02:
            lines.append("The score is suspiciously high. Check for a column that gives the answer away.")
    top = res["importance"][res["importance"]["importance"] > 0].head(3)["column"].tolist()
    if top:
        lines.append("The columns that matter most: " + ", ".join(f"'{c}'" for c in top) + ".")
    return lines


def run_clusters(df, cols, k):
    X = SimpleImputer(strategy="median").fit_transform(df[cols])
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    labels = km.labels_
    sil = float(silhouette_score(Xs, labels, sample_size=min(5000, len(Xs)), random_state=42))
    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
    out = df.copy()
    out["segment"] = ["Segment " + str(int(x) + 1) for x in labels]
    profile = out.groupby("segment")[cols].mean().round(2)
    profile.insert(0, "rows", out.groupby("segment").size())
    plot = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "segment": out["segment"].values})
    if len(plot) > 3000:
        plot = plot.sample(3000, random_state=42)
    return {"data": out, "profile": profile, "plot": plot, "silhouette": sil, "k": k, "cols": cols}


# ---------- database (stage 5) ----------
DEFAULT_PORTS = {"MySQL": 3306, "PostgreSQL": 5432}
SAMPLE_DB = "deralytic_sample.db"
TABLE_NAME_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def make_engine(kind, p):
    if kind == "SQLite":
        return create_engine(URL.create("sqlite", database=p["path"]))
    driver = {"MySQL": "mysql+pymysql", "PostgreSQL": "postgresql+psycopg2"}[kind]
    url = URL.create(
        driver, username=p["user"], password=p["password"],
        host=p["host"], port=int(p["port"]), database=p["database"],
    )
    return create_engine(url, pool_pre_ping=True)


def make_sample_frames():
    rng = np.random.default_rng(42)
    n = 600
    regions = ["East", "West", "South", "Central"]
    cats = ["Furniture", "Office Supplies", "Technology"]
    discount = rng.choice([0, 0.1, 0.2, 0.4, 0.5], size=n, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    sales = rng.gamma(2.0, 150.0, size=n).round(2)
    profit = (sales * (0.25 - discount * 0.7) + rng.normal(0, 8, size=n)).round(2)
    sales_df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="12h").strftime("%Y-%m-%d"),
        "customer_id": rng.integers(1, 101, size=n),
        "region": rng.choice(regions, size=n),
        "category": rng.choice(cats, size=n),
        "sales": sales, "discount": discount, "profit": profit,
    })
    cust_df = pd.DataFrame({
        "customer_id": np.arange(1, 101),
        "segment": rng.choice(["Consumer", "Corporate", "Home Office"], size=100),
        "age": rng.integers(18, 70, size=100),
    })
    return sales_df, cust_df


def create_sample_db(path):
    sales_df, cust_df = make_sample_frames()
    engine = create_engine(URL.create("sqlite", database=path))
    sales_df.to_sql("sales", engine, index=False, if_exists="replace")
    cust_df.to_sql("customers", engine, index=False, if_exists="replace")
    return engine


def connect_and_list(engine):
    names = inspect(engine).get_table_names()
    return sorted(names)


def load_table(engine, table, limit):
    tbl = Table(table, MetaData(), autoload_with=engine)
    with engine.connect() as conn:
        return pd.read_sql_query(select(tbl).limit(int(limit)), conn)


def safe_query(engine, sql):
    q = sql.strip().rstrip(";").strip()
    if not q:
        raise ValueError("Type a query first.")
    if not q.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed here.")
    if ";" in q:
        raise ValueError("Run one query at a time (remove the extra semicolon).")
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql_query(text(q), conn)


def database_source():
    """Shows the database connection form. Returns (dataframe, source_key) or (None, None)."""
    kind = st.selectbox("Database type", ["SQLite", "MySQL", "PostgreSQL"], key="db_kind")
    st.caption("MySQL and PostgreSQL only work with databases that this app can reach. "
               "On the public web version, use a file upload, the sample data or the SQLite sample database.")
    engine = None
    if st.session_state.get("db_kind_connected") == kind:
        engine = st.session_state.get("db_engine")

    if kind == "SQLite":
        path = st.text_input("Full path to your .db file", key="db_path",
                             help="Example: C:\\Users\\YourName\\Desktop\\mydata.db")
        c1, c2 = st.columns(2)
        connect = c1.button("Connect", key="db_connect")
        sample = c2.button("Create a sample database to practise", key="db_sample")
        if sample:
            try:
                full = os.path.abspath(SAMPLE_DB)
                eng = create_sample_db(full)
                st.session_state.update(db_engine=eng, db_kind_connected=kind,
                                        db_tables=connect_and_list(eng))
                st.success(f"Sample database created at {full} and connected.")
                engine = eng
            except Exception as e:
                st.error(f"Could not create the sample database: {e}")
        elif connect:
            if not path.strip():
                st.warning("Enter the path to a .db file first.")
            elif not os.path.isfile(path.strip().strip('"')):
                st.error("That file was not found. Check the path, including the file name and extension.")
            else:
                try:
                    eng = make_engine(kind, {"path": path.strip().strip('"')})
                    tables = connect_and_list(eng)
                    st.session_state.update(db_engine=eng, db_kind_connected=kind, db_tables=tables)
                    engine = eng
                except Exception as e:
                    st.error(f"Could not open that file: {e}")
    else:
        a, b = st.columns(2)
        host = a.text_input("Host", "localhost", key="db_host")
        port = b.number_input("Port", 1, 65535, DEFAULT_PORTS[kind], key=f"db_port_{kind}")
        c, d = st.columns(2)
        user = c.text_input("Username", key="db_user")
        password = d.text_input("Password", type="password", key="db_pass")
        dbname = st.text_input("Database name", key="db_name")
        if st.button("Connect", key="db_connect"):
            if not (host and user and dbname):
                st.warning("Fill in the host, username and database name.")
            else:
                try:
                    eng = make_engine(kind, {"host": host, "port": port, "user": user,
                                             "password": password, "database": dbname})
                    tables = connect_and_list(eng)
                    st.session_state.update(db_engine=eng, db_kind_connected=kind, db_tables=tables)
                    engine = eng
                except ModuleNotFoundError as e:
                    pkg = "pymysql" if kind == "MySQL" else "psycopg2-binary"
                    st.error(f"The driver is missing ({e.name}). Run: pip install {pkg}")
                except Exception as e:
                    st.error(f"Could not connect. Check the host, port, username, password and database name. Details: {e}")

    if engine is None:
        return None, None

    tables = st.session_state.get("db_tables", [])
    st.success(f"Connected to {kind}. {len(tables)} table(s) found.")
    how = st.radio("How do you want to load data?", ["Pick a table", "Write a SQL query"], horizontal=True, key="db_how")
    if how == "Pick a table":
        if not tables:
            st.info("This database has no tables yet.")
        else:
            table = st.selectbox("Table", tables, key="db_table")
            limit = st.number_input("Maximum rows to load", 100, 1_000_000, 50_000, step=1000, key="db_limit")
            if st.button("Load table", key="db_load_table"):
                try:
                    st.session_state["db_raw"] = load_table(engine, table, limit)
                    st.session_state["db_label"] = f"{kind} table {table}"
                    st.session_state["db_load_id"] = st.session_state.get("db_load_id", 0) + 1
                except Exception as e:
                    st.error(f"Could not load the table: {e}")
    else:
        sql = st.text_area("SQL query (SELECT only)", "SELECT * FROM ", height=120, key="db_sql")
        if st.button("Run query", key="db_run"):
            try:
                st.session_state["db_raw"] = safe_query(engine, sql)
                st.session_state["db_label"] = f"{kind} query"
                st.session_state["db_load_id"] = st.session_state.get("db_load_id", 0) + 1
            except Exception as e:
                st.error(f"Could not run the query: {e}")

    data = st.session_state.get("db_raw")
    if data is None:
        st.info("Load a table or run a query to get started.")
        return None, None
    if data.empty:
        st.warning("That returned no rows.")
        return None, None
    st.caption(f"Loaded {len(data):,} rows and {data.shape[1]} columns from {st.session_state.get('db_label')}.")
    return data, f"db-{st.session_state.get('db_load_id', 0)}"


# ---------- upload / database ----------
source = st.radio("Where is your data?", ["Upload a file", "Connect to a database"], horizontal=True)
if source == "Upload a file":
    uploaded = st.file_uploader("Upload your data", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        st.session_state["use_sample"] = False
        try:
            raw = read_file(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            footer()
            st.stop()
        source_key = f"file-{uploaded.name}-{uploaded.size}"
    else:
        if st.button("Or try it with sample sales data"):
            st.session_state["use_sample"] = True
        if not st.session_state.get("use_sample"):
            st.info("Upload a file or click the sample data button to get started.")
            footer()
            st.stop()
        raw = make_sample_frames()[0]
        source_key = "sample-sales"
        st.success("Using sample sales data (600 rows). Upload your own file any time to replace it.")
else:
    raw, source_key = database_source()
    if raw is None:
        footer()
        st.stop()

# ---------- cleaning options ----------
st.sidebar.header("Cleaning options")
opts = {
    "names": st.sidebar.checkbox("Standardise column names", True),
    "strip": st.sidebar.checkbox("Trim spaces in text", True),
    "types": st.sidebar.checkbox("Fix data types (numbers and dates)", True),
    "dupes": st.sidebar.checkbox("Remove duplicate rows", True),
    "num_fill": st.sidebar.selectbox("Missing numbers", ["Median", "Mean", "Zero", "Drop rows", "Leave as is"]),
    "cat_fill": st.sidebar.selectbox("Missing text", ["Unknown", "Most common value", "Leave as is"]),
    "outliers": st.sidebar.checkbox("Cap outliers (IQR rule)", False),
}
base, log = clean(raw, opts)

file_key = source_key
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state["steps"] = []
    st.session_state.pop("report", None)
    st.session_state.pop("ml_result", None)
    st.session_state.pop("cluster_result", None)
steps = st.session_state["steps"]

df, step_warnings = apply_steps(base, steps)
log = log + [describe_step(x) for x in steps]

tab1, tab_cols, tab2, tab3, tab4, tab5, tab_ml, tab_rep = st.tabs(
    ["Overview", "Clean columns", "Cleaned data", "Analysis", "Charts", "Insights", "Predict", "Report"]
)

# ---------- overview ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{len(raw):,}")
    c2.metric("Rows (cleaned)", f"{len(df):,}")
    c3.metric("Columns", df.shape[1])
    c4.metric("Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}")
    st.subheader("What was cleaned")
    for item in log or ["No cleaning steps selected."]:
        st.write("- " + item)
    st.subheader("Missing values in the raw data")
    miss = raw.isna().sum().rename("missing").to_frame()
    miss["percent"] = (miss["missing"] / len(raw) * 100).round(1)
    st.dataframe(miss[miss["missing"] > 0] if miss["missing"].any() else miss.head(0), width="stretch")
    st.subheader("Column types after cleaning")
    st.dataframe(df.dtypes.astype(str).rename("type").to_frame(), width="stretch")


# ---------- clean columns (stage 2) ----------
with tab_cols:
    st.subheader("Clean one column at a time")
    for w in step_warnings:
        st.warning(w)
    if len(df.columns) == 0:
        st.info("No columns left.")
    else:
        col = st.selectbox("Column", list(df.columns), key="col_pick")
        series = df[col]
        m1, m2, m3 = st.columns(3)
        m1.metric("Type", str(series.dtype))
        m2.metric("Missing", int(series.isna().sum()))
        m3.metric("Unique values", int(series.nunique()))
        st.caption("Sample values: " + ", ".join(series.dropna().astype(str).unique()[:6]))

        action = st.selectbox(
            "What do you want to do?",
            ["Fill missing values", "Replace a value", "Change type", "Rename column",
             "Drop rows where missing", "Drop column"],
            key="action_pick",
        )
        new_step = None
        if action == "Fill missing values":
            method = st.selectbox("Fill with", ["Median", "Mean", "Zero", "Most common value", "Custom value"])
            value = st.text_input("Custom value") if method == "Custom value" else ""
            if st.button("Apply", key="apply_fill"):
                new_step = {"type": "fill", "col": col, "method": method, "value": value}
        elif action == "Replace a value":
            old = st.text_input("Value to find (exactly as shown)")
            new = st.text_input("Replace with")
            if st.button("Apply", key="apply_replace") and old != "":
                new_step = {"type": "replace", "col": col, "old": old, "new": new}
        elif action == "Change type":
            to = st.selectbox("Change to", ["Text", "Number", "Date"])
            if st.button("Apply", key="apply_type"):
                new_step = {"type": "to_type", "col": col, "to": to}
        elif action == "Rename column":
            new = st.text_input("New name")
            if st.button("Apply", key="apply_rename") and new.strip():
                new_step = {"type": "rename", "col": col, "new": new.strip()}
        elif action == "Drop rows where missing":
            if st.button("Apply", key="apply_dropmissing"):
                new_step = {"type": "drop_missing", "col": col}
        elif action == "Drop column":
            if st.button("Apply", key="apply_drop"):
                new_step = {"type": "drop", "col": col}

        if new_step:
            steps.append(new_step)
            st.rerun()

    st.divider()
    st.subheader("Your cleaning steps")
    if steps:
        for i, x in enumerate(steps, 1):
            st.write(f"{i}. {describe_step(x)}")
        b1, b2 = st.columns(2)
        if b1.button("Undo last step"):
            steps.pop()
            st.rerun()
        if b2.button("Reset all steps"):
            steps.clear()
            st.rerun()
    else:
        st.caption("No column-level steps yet. Anything you apply here is listed, and you can undo it.")

# ---------- cleaned data + download ----------
with tab2:
    st.dataframe(df, width="stretch")
    st.download_button("Download cleaned CSV", df.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned")
    st.download_button(
        "Download cleaned Excel", buf.getvalue(), "cleaned_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()
    st.subheader("Save cleaned data to a database")
    eng = st.session_state.get("db_engine")
    if eng is None:
        st.caption("Connect to a database (choose 'Connect to a database' at the top) to save your cleaned data as a table.")
    else:
        new_table = st.text_input("New table name", "cleaned_data", key="save_table")
        overwrite = st.checkbox("Replace the table if it already exists", False, key="save_overwrite")
        if overwrite:
            st.warning("Replacing deletes the existing table with that name and its data.")
        if st.button("Save to database", key="save_btn"):
            if not TABLE_NAME_OK.match(new_table or ""):
                st.error("Use letters, numbers and underscores only, and do not start with a number.")
            else:
                try:
                    df.to_sql(new_table, eng, index=False, if_exists="replace" if overwrite else "fail")
                    st.session_state["db_tables"] = connect_and_list(eng)
                    st.success(f"Saved {len(df):,} rows to the table '{new_table}'.")
                except ValueError:
                    st.error(f"A table named '{new_table}' already exists. Pick another name or tick the replace box.")
                except Exception as e:
                    st.error(f"Could not save the table: {e}")

num_cols = list(df.select_dtypes(include="number").columns)
cat_cols = list(df.select_dtypes(exclude=["number", "datetime"]).columns)

# ---------- analysis ----------
with tab3:
    if num_cols:
        st.subheader("Summary statistics")
        st.dataframe(df[num_cols].describe().T, width="stretch")
    if len(num_cols) >= 2:
        st.subheader("Correlation heatmap")
        fig = px.imshow(df[num_cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")
    if cat_cols and num_cols:
        st.subheader("Group analysis")
        g1, g2, g3 = st.columns(3)
        by = g1.selectbox("Group by", cat_cols)
        val = g2.selectbox("Measure", num_cols)
        agg = g3.selectbox("Calculation", ["sum", "mean", "median", "count"])
        grouped = df.groupby(by)[val].agg(agg).sort_values(ascending=False).reset_index()
        st.dataframe(grouped, width="stretch")
        st.plotly_chart(px.bar(grouped.head(20), x=by, y=val), width="stretch")

# ---------- charts ----------
with tab4:
    kind = st.selectbox("Chart type", ["Histogram", "Box plot", "Bar (average)", "Scatter", "Line"])
    cols = list(df.columns)
    if kind in ("Histogram", "Box plot") and num_cols:
        x = st.selectbox("Column", num_cols)
        fig = px.histogram(df, x=x) if kind == "Histogram" else px.box(df, y=x)
        st.plotly_chart(fig, width="stretch")
    elif kind == "Bar (average)" and cat_cols and num_cols:
        x = st.selectbox("Category", cat_cols)
        y = st.selectbox("Value", num_cols)
        st.plotly_chart(px.bar(df.groupby(x)[y].mean().reset_index(), x=x, y=y), width="stretch")
    elif kind in ("Scatter", "Line"):
        x = st.selectbox("X axis", cols)
        y = st.selectbox("Y axis", num_cols) if num_cols else None
        if y:
            fig = px.scatter(df, x=x, y=y) if kind == "Scatter" else px.line(df.sort_values(x), x=x, y=y)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("This chart needs a different mix of columns than your data has.")

# ---------- insights ----------
with tab5:
    for line in make_insights(df):
        st.write("- " + line)


# ---------- predict / segments (stage 4) ----------
with tab_ml:
    mode = st.radio("What do you want to do?", ["Predict a column", "Find segments (clustering)"], horizontal=True)

    if mode == "Predict a column":
        st.caption("Pick a column to predict. Deralytic trains a model on part of your data and tests it on the rest.")
        targets = [
            c for c in df.columns
            if not pd.api.types.is_datetime64_any_dtype(df[c]) and not looks_like_id(c)
            and df[c].nunique() > 1 and (pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 20)
        ]
        if not targets:
            st.info("No suitable column to predict. A target needs to be a number or a column with up to 20 categories.")
        else:
            target = st.selectbox("Column to predict", targets, key="ml_target")
            options = candidate_features(df, target)
            feats = st.multiselect("Columns to learn from", options, default=options, key=f"ml_feats_{target}")
            test_size = st.slider("Share of rows kept for testing", 10, 40, 25, step=5) / 100
            if st.button("Train model", key="train_btn"):
                if not feats:
                    st.warning("Pick at least one column to learn from.")
                else:
                    with st.spinner("Training..."):
                        try:
                            st.session_state["ml_result"] = train_model(df, target, feats, test_size)
                        except Exception as e:
                            st.error(f"Could not train the model: {e}")

            res = st.session_state.get("ml_result")
            if res and res["target"] == target:
                st.subheader("Results")
                if res["task"] == "regression":
                    a, b, c_ = st.columns(3)
                    a.metric("R-squared", f"{res['r2']:.2f}")
                    b.metric("Average error", f"{res['mae']:,.2f}")
                    c_.metric("Typical error (RMSE)", f"{res['rmse']:,.2f}")
                else:
                    a, b, c_ = st.columns(3)
                    a.metric("Accuracy", f"{res['accuracy']:.0%}")
                    b.metric("Guessing baseline", f"{res['baseline']:.0%}")
                    c_.metric("F1 score", f"{res['f1']:.2f}")
                for line in explain_result(res):
                    st.write("- " + line)
                st.caption(f"Trained on {res['n_train']:,} rows and tested on {res['n_test']:,} rows.")

                imp = res["importance"].head(10).iloc[::-1]
                st.plotly_chart(px.bar(imp, x="importance", y="column", orientation="h",
                                       title="Which columns matter most"), width="stretch")
                st.caption("Importance shows how much the score drops when a column is shuffled.")
                if res["task"] == "regression":
                    comp = pd.DataFrame({"actual": res["y_test"], "predicted": res["pred"]}).head(2000)
                    st.plotly_chart(px.scatter(comp, x="actual", y="predicted", title="Actual vs predicted"),
                                    width="stretch")
                else:
                    cm = pd.DataFrame(res["cm"], index=res["labels"], columns=res["labels"])
                    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                              labels=dict(x="Predicted", y="Actual"),
                                              title="Confusion matrix"), width="stretch")

    else:
        st.caption("Group similar rows into segments, for example to find customer groups.")
        seg_options = [c for c in num_cols if not looks_like_id(c) and df[c].notna().any() and df[c].nunique() > 1]
        if len(seg_options) < 2:
            st.info("Segments need at least two numeric columns.")
        else:
            seg_cols = st.multiselect("Columns to group by", seg_options, default=seg_options[:5], key="seg_cols")
            k = st.slider("Number of segments", 2, 8, 3)
            if st.button("Find segments", key="seg_btn"):
                if len(seg_cols) < 2:
                    st.warning("Pick at least two columns.")
                elif len(df) < k * 5:
                    st.warning("Not enough rows for that many segments.")
                else:
                    with st.spinner("Finding segments..."):
                        try:
                            st.session_state["cluster_result"] = run_clusters(df, seg_cols, k)
                        except Exception as e:
                            st.error(f"Could not find segments: {e}")
            cr = st.session_state.get("cluster_result")
            if cr:
                sil = cr["silhouette"]
                quality = "clearly separated" if sil > 0.5 else "moderately separated" if sil > 0.25 else "overlapping"
                st.write(f"The {cr['k']} segments are **{quality}** (silhouette score {sil:.2f}).")
                st.subheader("What each segment looks like (averages)")
                st.dataframe(cr["profile"], width="stretch")
                st.plotly_chart(px.scatter(cr["plot"], x="x", y="y", color="segment",
                                           title="Segments (2D view of your columns)"), width="stretch")
                st.download_button("Download data with segments", cr["data"].to_csv(index=False).encode("utf-8"),
                                   "segmented_data.csv", "text/csv")

# ---------- report ----------
with tab_rep:
    st.subheader("One-click report")
    st.caption("Creates a PDF with an overview, what was cleaned, statistics, insights and charts.")
    rep_title = st.text_input("Report title", "Data Analysis Report")
    rep_charts = st.checkbox("Include charts", True)
    if st.button("Create report"):
        with st.spinner("Building your report..."):
            st.session_state["report"] = build_report(rep_title, raw, df, log, make_insights(df), rep_charts)
    if st.session_state.get("report"):
        st.success("Your report is ready.")
        st.download_button("Download PDF report", st.session_state["report"], "analysis_report.pdf", "application/pdf")

footer()
