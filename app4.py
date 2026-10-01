import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:  # the app still runs; the chat tab explains what is missing
    OpenAI = None

# ==============================================================================
# 1. PAGE CONFIGURATION & GEOGRAPHIC UTILITIES
# ==============================================================================
st.set_page_config(
    page_title="2025 CDC Provisional Natality Explorer",
    page_icon="👶",
    layout="wide"
)

# Mapping full US State and District names to 2-letter postal codes for Plotly choropleth rendering
STATE_TO_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

REQUIRED_COLUMNS = ['state_of_residence', 'month', 'month_code', 'year_code', 'sex_of_infant', 'births']

# --- AI assistant constants -------------------------------------------------------
# Provider detection by API-key prefix.
LLM_PROVIDERS = {
    "gsk_": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
    "xai-": {
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3-mini",
    },
}

# Secret names accepted for the API key (first one found is used).
API_KEY_SECRET_NAMES = ["GROQ_API_KEY", "GROK_API_KEY", "XAI_API_KEY", "LLM_API_KEY"]

# Groq models tried, in order, if the preferred model is retired or unavailable.
GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
]

# Substrings (lower-case) that identify a "model unavailable" API error.
MODEL_UNAVAILABLE_PHRASES = [
    "not found",
    "decommissioned",
    "deprecated",
    "does not exist",
    "not available",
    "model_not_found",
    "model_decommissioned",
]

SUGGESTED_QUESTIONS = [
    "Which 3 states had the most births?",
    "Which month had the fewest births, and why might that be?",
    "What is the male-to-female ratio in the current selection?",
]

MAX_HISTORY_MESSAGES = 6   # only the most recent messages are sent to the model

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2000      # gpt-oss is a reasoning model and spends tokens "thinking"

EMPTY_ANSWER_MESSAGE = "The model returned an empty answer. Please try rephrasing your question."

SYSTEM_PROMPT = """You are a friendly data assistant for the CDC/NCHS provisional 2025 U.S. natality (births) data.

Rules:
- Answer ONLY from the DATA SUMMARY below. It reflects the user's current sidebar filters.
- All values are raw birth COUNTS, not birth rates. When comparing states, remind the user that population size drives the counts.
- The data is provisional and may be revised.
- If a question cannot be answered from the data (for example race, mother's age, or other years), say so and suggest what data would be needed.
- Use thousands separators, double-check your arithmetic, and be concise."""

# ==============================================================================
# 2. DATA LOADING & VALIDATION
# ==============================================================================
@st.cache_data
def load_and_validate_data(file_path: str = "Provisional_Natality_2025_CDC.csv") -> pd.DataFrame:
    """Loads dataset, executes data validation checks, sets categorical month ordering,
    and maps state postal codes for map rendering."""
    if not os.path.exists(file_path):
        st.error(f"Dataset missing at location: {file_path}. Please place 'Provisional_Natality_2025_CDC.csv' in the root directory.")
        st.stop()
        
    df = pd.read_csv(file_path)
    
    # Validation Check 1: Required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Data Validation Error: Missing required columns: {missing_cols}")
        st.stop()
        
    # Validation Check 2: Non-negative birth counts
    if (df['births'] < 0).any():
        st.error("Data Validation Error: Negative values detected in 'births' column.")
        st.stop()

    # Validation Check 3: Check for null values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        st.warning("Notice: Null values found in dataset. Rows with nulls in key fields will be dropped.")
        df = df.dropna(subset=REQUIRED_COLUMNS)

    # Convert month column to Categorical to preserve chronological ordering
    df['month'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
    
    # Add state 2-letter code mapping
    df['state_abbr'] = df['state_of_residence'].map(STATE_TO_ABBR)
    
    return df

# ==============================================================================
# 3. SIDEBAR FILTERS COMPONENT
# ==============================================================================
def render_sidebar(df):
    """Renders sidebar filters, action buttons, and active selection summaries."""
    st.sidebar.header("Filter Options")
    
    all_states = sorted(df['state_of_residence'].unique().tolist())
    all_months = [m for m in df['month'].cat.categories if m in df['month'].unique()]
    all_sexes = ['All', 'Female', 'Male']
    
    # Initialize session state for filter selections if not set
    if 'selected_states' not in st.session_state:
        st.session_state.selected_states = all_states
    if 'selected_months' not in st.session_state:
        st.session_state.selected_months = all_months
    if 'selected_sex' not in st.session_state:
        st.session_state.selected_sex = 'All'

    # Action Buttons: Select All & Reset
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All"):
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = 'All'
        st.rerun()

    if col2.button("Reset Filters"):
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = 'All'
        st.rerun()

    # Multiselect Inputs
    selected_states = st.sidebar.multiselect(
        "Select State/Geography:",
        options=all_states,
        default=st.session_state.selected_states
    )
    
    selected_months = st.sidebar.multiselect(
        "Select Month:",
        options=all_months,
        default=st.session_state.selected_months
    )

    selected_sex = st.sidebar.selectbox(
        "Select Infant Sex:",
        options=all_sexes,
        index=all_sexes.index(st.session_state.selected_sex)
    )

    # Sync back to session state
    st.session_state.selected_states = selected_states
    st.session_state.selected_months = selected_months
    st.session_state.selected_sex = selected_sex

    # Active Filters Summary Box
    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Filter Summary")
    st.sidebar.info(
        f"**Geographies Selected:** {len(selected_states)} of {len(all_states)}\n\n"
        f"**Months Selected:** {len(selected_months)} of {len(all_months)}\n\n"
        f"**Infant Sex Selection:** {selected_sex}"
    )

    return selected_states, selected_months, selected_sex

# ==============================================================================
# 4. KPI CARDS COMPONENT
# ==============================================================================
def render_kpi_cards(filtered_df):
    """Computes and displays top-level key performance metrics."""
    if filtered_df.empty:
        return

    total_births = filtered_df['births'].sum()
    num_states = filtered_df['state_of_residence'].nunique()
    
    # Average births per selected month
    month_counts = filtered_df.groupby('month', observed=True)['births'].sum()
    avg_births_per_month = month_counts.mean() if not month_counts.empty else 0
    top_month = month_counts.idxmax() if not month_counts.empty else "N/A"
    
    # Geography with highest birth count
    state_counts = filtered_df.groupby('state_of_residence', observed=True)['births'].sum()
    top_state = state_counts.idxmax() if not state_counts.empty else "N/A"

    # Display KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Total Births", f"{total_births:,.0f}")
    kpi2.metric("Selected Geographies", f"{num_states:,}")
    kpi3.metric("Avg Births / Month", f"{avg_births_per_month:,.0f}")
    kpi4.metric("Top Geography", f"{top_state}")
    kpi5.metric("Top Month", f"{top_month}")

# ==============================================================================
# 5. CHARTS AND VISUALIZATIONS
# ==============================================================================
def plot_monthly_trend(df):
    """Generates monthly birth trend line chart (non-truncated axes)."""
    monthly_data = df.groupby('month', observed=True)['births'].sum().reset_index()
    fig = px.line(
        monthly_data, 
        x='month', 
        y='births', 
        markers=True,
        title="Total Births by Month (Chronological)",
        labels={'month': 'Month', 'births': 'Birth Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_yaxes(rangemode='tozero')
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Births: %{y:,}<extra></extra>")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_choropleth(df):
    """Renders US state-level choropleth map."""
    state_data = df.groupby(['state_abbr', 'state_of_residence'], observed=True)['births'].sum().reset_index()
    fig = px.choropleth(
        state_data,
        locations='state_abbr',
        locationmode="USA-states",
        color='births',
        scope="usa",
        hover_name='state_of_residence',
        color_continuous_scale="Viridis",
        title="Birth Counts by US State",
        labels={'births': 'Total Births'}
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Total Births: %{z:,}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_state_ranking(df):
    """Renders horizontal bar chart of top/bottom state birth totals."""
    state_data = df.groupby('state_of_residence', observed=True)['births'].sum().reset_index()
    state_data = state_data.sort_values(by='births', ascending=True)
    
    fig = px.bar(
        state_data, 
        y='state_of_residence', 
        x='births', 
        orientation='h',
        title="State Birth Count Rankings",
        labels={'state_of_residence': 'State', 'births': 'Birth Count'},
        color_discrete_sequence=['#2ca02c']
    )
    fig.update_xaxes(rangemode='tozero')
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,}<extra></extra>")
    fig.update_layout(height=max(400, len(state_data) * 20), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_top_bottom_comparison(df, n=5):
    """Compares top N vs bottom N geographies in current selection."""
    state_totals = df.groupby('state_of_residence', observed=True)['births'].sum().sort_values(ascending=False)
    if len(state_totals) < 2:
        return None

    top_states = state_totals.head(n)
    bottom_states = state_totals.tail(n).iloc[::-1]

    comp_df = pd.concat([
        pd.DataFrame({'State': top_states.index, 'Births': top_states.values, 'Group': f'Top {n}'}),
        pd.DataFrame({'State': bottom_states.index, 'Births': bottom_states.values, 'Group': f'Bottom {n}'})
    ])

    fig = px.bar(
        comp_df,
        x='State',
        y='Births',
        color='Group',
        barmode='group',
        title=f"Comparison: Top {n} vs Bottom {n} Geographies",
        color_discrete_sequence=['#1f77b4', '#d62728']
    )
    fig.update_yaxes(rangemode='tozero')
    fig.update_traces(hovertemplate="<b>%{x}</b> (%{fullData.name})<br>Births: %{y:,}<extra></extra>")
    return fig

def plot_state_month_heatmap(df):
    """Renders state-by-month birth count heatmap."""
    pivot = df.pivot_table(index='state_of_residence', columns='month', values='births', aggfunc='sum', observed=True).fillna(0)
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="State", color="Birth Count"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="YlGnBu",
        title="State vs Month Birth Count Heatmap"
    )
    fig.update_layout(height=max(400, len(pivot) * 18))
    return fig

def plot_sex_comparison(df):
    """Renders grouped bar chart comparing female and male birth counts by month."""
    sex_month = df.groupby(['month', 'sex_of_infant'], observed=True)['births'].sum().reset_index()
    fig = px.bar(
        sex_month,
        x='month',
        y='births',
        color='sex_of_infant',
        barmode='group',
        title="Monthly Birth Count Comparison by Infant Sex",
        labels={'month': 'Month', 'births': 'Birth Count', 'sex_of_infant': 'Infant Sex'},
        color_discrete_sequence=['#e377c2', '#1f77b4']
    )
    fig.update_yaxes(rangemode='tozero')
    fig.update_traces(hovertemplate="<b>%{x}</b> (%{fullData.name})<br>Births: %{y:,}<extra></extra>")
    return fig

# ==============================================================================
# 6. AI DATA ASSISTANT (CHATBOT)
# ==============================================================================
class AllModelsUnavailableError(Exception):
    """Raised when every configured model is retired, unknown or unavailable."""


def get_api_key() -> Optional[str]:
    """Returns the first API key found in Streamlit secrets, or None."""
    for name in API_KEY_SECRET_NAMES:
        try:
            value = st.secrets[name]
        except Exception:
            continue
        if value and str(value).strip():
            return str(value).strip()
    return None


def get_provider(api_key: str) -> Optional[dict]:
    """Detects the LLM provider from the API-key prefix."""
    for prefix, provider in LLM_PROVIDERS.items():
        if api_key.startswith(prefix):
            return provider
    return None


def get_model_override() -> Optional[str]:
    """Returns the optional LLM_MODEL secret, if present."""
    try:
        value = st.secrets["LLM_MODEL"]
    except Exception:
        return None
    return str(value).strip() if value and str(value).strip() else None


def get_candidate_models(provider: dict) -> list:
    """Builds the ordered list of models to try (the model that worked last goes first)."""
    preferred = get_model_override() or provider["default_model"]
    candidates = [preferred]
    if provider["name"] == "Groq":
        candidates += [m for m in GROQ_FALLBACK_MODELS if m != preferred]

    active = st.session_state.get("active_model")
    if active in candidates:
        candidates.remove(active)
        candidates.insert(0, active)
    return candidates


def is_model_unavailable_error(err: Exception) -> bool:
    """True only for errors that mean the model is retired, unknown or unavailable."""
    message = str(err).lower()
    return any(phrase in message for phrase in MODEL_UNAVAILABLE_PHRASES)


def build_data_summary(df: pd.DataFrame) -> str:
    """Builds a compact text summary of the FILTERED data (never the raw rows)."""
    sexes = sorted(df['sex_of_infant'].unique().tolist())
    sex_label = "All (Female and Male)" if len(sexes) > 1 else sexes[0]
    total = int(df['births'].sum())

    lines = [
        "DATA SUMMARY (matches the user's current sidebar filters)",
        "",
        "ACTIVE FILTERS:",
        f"- Infant sex: {sex_label}",
        f"- Geographies selected: {df['state_of_residence'].nunique():,}",
        f"- Months selected: {df['month'].nunique():,}",
        "",
        f"TOTAL BIRTHS: {total:,}",
        "",
        "BIRTHS BY SEX:",
    ]
    by_sex = df.groupby('sex_of_infant', observed=True)['births'].sum()
    lines += [f"- {sex}: {int(count):,}" for sex, count in by_sex.items()]

    lines += ["", "BIRTHS BY MONTH:"]
    by_month = df.groupby('month', observed=True)['births'].sum()
    lines += [f"- {month}: {int(count):,}" for month, count in by_month.items()]

    state_sex = (
        df.groupby(['state_of_residence', 'sex_of_infant'], observed=True)['births']
        .sum()
        .unstack(fill_value=0)
    )
    state_sex['Total'] = state_sex.sum(axis=1)
    state_sex = state_sex.sort_values('Total', ascending=False)
    sex_cols = [c for c in ['Female', 'Male'] if c in state_sex.columns]

    lines += ["", "BIRTHS BY STATE, ranked high to low (" + " | ".join(["State", "Total"] + sex_cols) + "):"]
    for state, row in state_sex.iterrows():
        parts = [str(state), f"{int(row['Total']):,}"] + [f"{int(row[c]):,}" for c in sex_cols]
        lines.append("- " + " | ".join(parts))

    return "\n".join(lines)


def create_stream_with_fallback(client, provider: dict, messages: list):
    """Starts a streaming completion, falling back to other models ONLY when the
    current model is unavailable. Any other error is raised normally."""
    for model in get_candidate_models(provider):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )
        if "gpt-oss" in model:
            # extra_body works with any openai library version
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception as err:
            if is_model_unavailable_error(err):
                continue
            raise
        st.session_state["active_model"] = model
        return stream
    raise AllModelsUnavailableError("No configured model is available.")


def stream_text(stream):
    """Yields only the text deltas from a streaming response."""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def friendly_error_message(err: Exception) -> str:
    """Translates API errors into short, friendly messages."""
    if isinstance(err, AllModelsUnavailableError):
        return (
            "None of the configured AI models are available. Add a LLM_MODEL secret "
            "in Streamlit Cloud (⋮ → Settings → Secrets) with a model name that your "
            "provider currently supports."
        )
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    if status == 401 or "invalid api key" in text or "invalid_api_key" in text or "error code: 401" in text:
        return "The API key was rejected. Please check the key saved in your Streamlit Secrets."
    if status == 429 or "rate limit" in text or "rate_limit" in text or "error code: 429" in text:
        return "The free-tier rate limit was reached. Please wait a minute and try again."
    return f"Sorry, something went wrong contacting the AI service: {err}"


def render_chatbot(filtered_df: pd.DataFrame):
    """Renders the AI chat tab, grounded in the currently filtered data."""
    st.subheader("Ask the Data Assistant")

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "**No API key found.** To enable the assistant, add your key to Streamlit Cloud: "
            "open your app, click **⋮ → Settings → Secrets**, and add a line like "
            "`GROQ_API_KEY = \"gsk_your_key_here\"`, then save."
        )
        return

    provider = get_provider(api_key)
    if provider is None:
        st.warning(
            "The API key format is not recognized. Groq keys start with `gsk_` and xAI keys "
            "start with `xai-`. Please check the key saved in **⋮ → Settings → Secrets**."
        )
        return

    if OpenAI is None:
        st.error("The 'openai' package is not installed. Add `openai>=1.40.0` to requirements.txt.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = get_model_override() or provider["default_model"]

    caption_slot = st.empty()

    def show_caption():
        caption_slot.caption(
            f"Powered by {provider['name']} · model {st.session_state['active_model']}. "
            "Answers are based on the data matching your current sidebar filters. "
            "AI can make mistakes — verify key numbers with the charts."
        )

    show_caption()

    # Suggested questions + clear button
    pending_question = None
    cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[i].button(question, key=f"suggested_{i}", width="stretch"):
            pending_question = question
    if cols[-1].button("🗑️ Clear chat", key="clear_chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input("Ask a question about the births data…")
    user_question = pending_question or typed_question
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=api_key, base_url=provider["base_url"], timeout=60.0)
            system_message = {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\n" + build_data_summary(filtered_df),
            }
            recent = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
            stream = create_stream_with_fallback(client, provider, [system_message] + recent)
            answer = st.write_stream(stream_text(stream))
            if not isinstance(answer, str) or not answer.strip():
                answer = EMPTY_ANSWER_MESSAGE
                st.markdown(answer)
        except Exception as err:
            answer = friendly_error_message(err)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    show_caption()

# ==============================================================================
# 7. TAB RENDERING
# ==============================================================================
def render_overview_tab(filtered_df):
    st.subheader("Overview & Macro Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_choropleth(filtered_df), width="stretch")
    with col2:
        st.plotly_chart(plot_monthly_trend(filtered_df), width="stretch")

def render_geo_tab(filtered_df):
    st.subheader("Geographic Distribution Analysis")
    top_bot_fig = plot_top_bottom_comparison(filtered_df)
    if top_bot_fig:
        st.plotly_chart(top_bot_fig, width="stretch")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(plot_state_ranking(filtered_df), width="stretch")
    with col2:
        st.plotly_chart(plot_state_month_heatmap(filtered_df), width="stretch")

def render_monthly_sex_tab(filtered_df):
    st.subheader("Monthly and Sex-Based Patterns")
    st.plotly_chart(plot_sex_comparison(filtered_df), width="stretch")

def render_data_tab(filtered_df):
    st.subheader("Data Explorer & Export")
    
    # Search input filter for table
    search_term = st.text_input("Search state within current filtered table:", "")
    display_df = filtered_df.copy()
    
    if search_term:
        display_df = display_df[display_df['state_of_residence'].str.contains(search_term, case=False, na=False)]
        
    st.dataframe(
        display_df.style.format({'births': '{:,}'}),
        width="stretch",
        hide_index=True
    )
    
    # Download button for CSV export
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_cdc_natality_2025.csv",
        mime="text/csv"
    )

def render_about_tab():
    st.subheader("About the CDC Natality Dataset")
    st.markdown("""
    ### Data Context & Methodology
    * **Source:** Centers for Disease Control and Prevention (CDC) - National Center for Health Statistics (NCHS)[cite: 1].
    * **Data Type:** Provisional Natality Figures (2025)[cite: 1].
    * **Key Distinction:** This dataset reports raw **birth counts**, not **birth rates**. Birth counts reflect total volume, whereas birth rates adjust for population size per geography.
    * **Provisional Status:** Provisional counts are subject to revision as additional records are finalized by reporting jurisdictions[cite: 1].
    
    ### Educational Objectives for Business Analytics
    * Visualizing time-series trends without zero-axis truncation.
    * Exploring geographic disparity using interactive choropleth mapping.
    * Dynamic aggregation and filtering across tabular variables.
    """)

# ==============================================================================
# 8. MAIN APPLICATION EXECUTION
# ==============================================================================
def main():
    # Header Section
    st.title("2025 CDC Provisional Natality Dashboard")
    st.caption("Interactive analysis of provisional US birth counts by geography, month, and infant sex.")
    
    st.warning(
        "**Important Notice:** Data displayed are **provisional figures** provided by the **CDC/NCHS**[cite: 1]. "
        "All numbers represent raw **birth counts**, NOT birth rates."
    )

    # Load & Validate Data
    df = load_and_validate_data()

    # Render Sidebar Filters
    selected_states, selected_months, selected_sex = render_sidebar(df)

    # Apply Filters
    filtered_df = df[
        (df['state_of_residence'].isin(selected_states)) &
        (df['month'].isin(selected_months))
    ]
    
    if selected_sex != 'All':
        filtered_df = filtered_df[filtered_df['sex_of_infant'] == selected_sex]

    # Handle Empty Filter Selection
    if filtered_df.empty:
        st.error("⚠️ No observations match the current filter selection. Please adjust your filters in the sidebar.")
        return

    # Render Top KPI Cards
    render_kpi_cards(filtered_df)
    st.markdown("---")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Geographic Analysis",
        "Monthly & Sex Analysis",
        "🤖 Ask the Data (AI)",
        "Data Table & Download",
        "About the Data"
    ])

    with tab1:
        render_overview_tab(filtered_df)

    with tab2:
        render_geo_tab(filtered_df)

    with tab3:
        render_monthly_sex_tab(filtered_df)

    with tab4:
        render_chatbot(filtered_df)

    with tab5:
        render_data_tab(filtered_df)

    with tab6:
        render_about_tab()

if __name__ == "__main__":
    main()
