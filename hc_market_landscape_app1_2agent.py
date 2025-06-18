import streamlit as st
import pandas as pd
import os
import plotly.express as px
from functools import reduce

from dotenv import load_dotenv
import os

# Load environment variables.
load_dotenv()

import streamlit as st
import pandas as pd
import os
from functools import reduce
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import json

st.set_page_config(page_title="Market Landscape & Share Analysis", layout="wide")
st.title("🏥 CALIBER360 | CA Healthcare Competitive Intelligence Dashboard")

# Gemini setup
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0)

def generate_insight(title, df_as, df_ed):
    if df_as.empty and df_ed.empty:
        return "No data available to generate insights."
    try:
        sample_as = df_as.sample(min(50, len(df_as)), random_state=42) if not df_as.empty else pd.DataFrame()
        sample_ed = df_ed.sample(min(50, len(df_ed)), random_state=42) if not df_ed.empty else pd.DataFrame()
        combined_df = pd.concat([sample_as, sample_ed]).dropna(how='all')
        sample = combined_df.to_dict(orient='records')

        prompt = f"""
        You are a confident and insightful healthcare analytics expert. 
        Based on the following dataset (sampled from user-selected facilities), 
        provide 2–3 clear and meaningful insights looking at the data pattern for 
        Ambulatory Surgery (AS) and Emergency Department (ED) across facilities. 
        Focus on trends you see within AS and ED, not between them. Do not include
        raw numbers in your insights. Most important, if more than one facility is selected, 
        then discuss the differences in volume and trend that stand out across them 
        (these facilities).  Do not get into demographic aspects when simply 
        addressing the AS and ED volume charts and data.  Do not say things like data lacks
        or data is missing, etc.

        When dealing with demographic data, focus on what is present rather than on
        what is not present.  Include in your insights specific ideas about differences
        across various groups within the demographic you're dealing with.  Focus on
        groups that show greater presence than others and reflect on why that might be.
        Most important, if more than one facility is selected, then discuss the differences in
        demographic representation that stand out across them (these facilities).  
        Do not say things like data lacks or data is missing, etc.

        Only provide insights based on data that is present in the dataset. 
        Do not comment on missing, incomplete, or insufficient data. 
        Assume that demographic and payer mix data is reliable and available 
        unless values are explicitly missing. Do not speculate about what 
        additional data might be helpful.  Do not say things like data lacks
        or data is missing, etc.

        Dataset:
        {json.dumps(sample)}
        """
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Insight generation failed: {str(e)}"
    

# Load data from AS and ED folders
@st.cache_data

def load_combined_data(folder_prefix):
    filenames = [
        f"../APP/combined_{folder_prefix}_2024_table{i}.csv" for i in range(7, 16)
    ]
    tables = []
    for file in filenames:
        path = os.path.join(".", file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.loc[:, ~df.columns.duplicated()]
            df['SOURCE'] = folder_prefix
            tables.append(df)
    return pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()

# Load both AS and ED data
df_as = load_combined_data("AS")
df_ed = load_combined_data("ED")

# Combine for filtering only
df_all = pd.concat([df_as, df_ed], axis=0, ignore_index=True)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
facilities = st.sidebar.multiselect(
    "Select Facility or Facilities",
    options=sorted(df_all['Facility'].dropna().unique())
)

# Helper function to render side-by-side comparison

def render_side_by_side(df_as, df_ed, group_col, value_label, chart_title):
    col1, col2 = st.columns(2)

    def safe_age_sort(val):
        val = str(val).strip()
        try:
            if '–' in val:
                return int(val.split('–')[0].strip())
            elif val[0].isdigit():
                return int(''.join([c for c in val if c.isdigit()]))
        except:
            pass
        return 9999  # catch-all for unparseable values like '80 Years +', 'Unknown'

    for source_df, label, container in zip(
        [df_as, df_ed],
        ["Ambulatory Surgery (AS)", "Emergency Department (ED)"],
        [col1, col2]
    ):
        with container:
            st.subheader(label)
            data = source_df[
                (source_df[group_col].notna()) &
                (~source_df.apply(lambda row: row.astype(str).str.upper().eq('TOTAL').any(), axis=1))
            ]
            data = data[data['Facility'].isin(facilities)] if facilities else data

            if not data.empty:
                data = data.rename(columns=lambda col: col.strip().upper().replace('"', ''))
                quarter_cols = [col for col in data.columns if 'Q1 2024' in col or 'Q2 2024' in col or 'Q3 2024' in col or 'Q4 2024' in col]
                data['TOTAL'] = data[quarter_cols].sum(axis=1)

                grouped = data.groupby(['FACILITY', group_col.upper()], as_index=False)['TOTAL'].sum()

                # Fix age formatting
                if group_col.upper() == 'AGE GROUPS':
                    grouped[group_col.upper()] = grouped[group_col.upper()].astype(str).str.replace('-', '–')
                    order = sorted(grouped[group_col.upper()].unique(), key=safe_age_sort)
                else:
                    grouped[group_col.upper()] = grouped[group_col.upper()].astype(str)
                    order = sorted(grouped[group_col.upper()].unique())

                grouped['PROPORTION'] = grouped.groupby('FACILITY')['TOTAL'].transform(lambda x: x / x.sum() * 100).round(1)

                fig = px.bar(
                    grouped,
                    y=group_col.upper(),
                    x='PROPORTION',
                    color='FACILITY',
                    barmode='group',
                    orientation='h',
                    text='PROPORTION',
                    title=f"{chart_title}",
                    category_orders={group_col.upper(): order}  # ✅ Valid usage here
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
                fig.update_layout(yaxis_title=group_col.title(), xaxis_title='Percentage (%)')
                st.plotly_chart(fig, use_container_width=True)


st.markdown("### 📉 Quarterly Encounter Trends by Facility: Ambulatory Surgery and Emergency Department")
col1, col2 = st.columns(2)

def render_quarterly_line_chart(df, title, container):
    with container:
        df = df[
            (~df.apply(lambda row: row.astype(str).str.upper().eq('TOTAL').any(), axis=1))
        ]
        df = df[df['Facility'].isin(facilities)] if facilities else df
        quarter_cols = [col for col in df.columns if 'Q1 2024' in col or 'Q2 2024' in col or 'Q3 2024' in col or 'Q4 2024' in col]

        if quarter_cols and not df.empty:
            trend_data = df[['Facility'] + quarter_cols].copy()
            trend_data = trend_data.groupby('Facility', as_index=False).first()
            trend_data = trend_data.melt(id_vars='Facility', var_name='Quarter', value_name='Encounters')
            trend_data['Quarter'] = pd.Categorical(
                trend_data['Quarter'],
                categories=['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
                ordered=True
            )
            trend_data = trend_data.sort_values(['Facility', 'Quarter'])

            fig = px.line(
                trend_data,
                x='Quarter',
                y='Encounters',
                color='Facility',
                markers=True,
                line_group='Facility',
                title=title
            )
            fig.update_layout(xaxis=dict(type='category'), yaxis_title='Encounters', xaxis_title='Quarter')
            st.plotly_chart(fig, use_container_width=True)

    

# ✅ Fixed: na=False avoids NaN error
render_quarterly_line_chart(df_as[df_as['Report'].str.contains('Ambulatory Surgery', case=False, na=False)], "AS Quarterly Trends", col1)

render_quarterly_line_chart(df_ed[df_ed['Report'].str.contains('Emergency Department', case=False, na=False)], "ED Quarterly Trends", col2)

cols_of_interest = ['Facility', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
filtered_as = df_as[df_as['Facility'].isin(facilities)][cols_of_interest] if facilities else df_as[cols_of_interest]
filtered_ed = df_ed[df_ed['Facility'].isin(facilities)][cols_of_interest] if facilities else df_ed[cols_of_interest]

if not filtered_as.empty or not filtered_ed.empty:
    st.markdown("#### 🤖 CALIBER360 Insights on Quarterly Encounter Trends")
    insight = generate_insight("Quarterly Encounter Trends", filtered_as, filtered_ed)
    st.info(insight)


# Charts to render
chart_specs = [
    ('AGE GROUPS', 'Age Distribution (%) by Facility'),
    ('SEX', 'Sex Distribution (%) by Facility'),
    ('ZIP CODE', 'California Residency (%) by Facility'),
    ('RACE', 'Race Distribution (%) by Facility'),
    ('ETHNICITY', 'Ethnicity Distribution (%) by Facility'),
    ('PREFERRED LANGUAGE SPOKEN_1', 'Preferred Language (%) by Facility'),
    ('DISPOSITION', 'Disposition Distribution (%) by Facility'),
    ('EXPECTED PAYER SOURCE', 'Expected Payer Source (%) by Facility'),
]

for colname, title in chart_specs:
    st.markdown(f"### {title}: AS vs ED")
    render_side_by_side(df_as, df_ed, colname, 'PROPORTION', title)

    # Limit to just facility + current demographic column
    cols_to_keep = ['Facility', colname]
    filtered_as = df_as[df_as['Facility'].isin(facilities)][cols_to_keep] if facilities else df_as[cols_to_keep]
    filtered_ed = df_ed[df_ed['Facility'].isin(facilities)][cols_to_keep] if facilities else df_ed[cols_to_keep]

    if not filtered_as.empty or not filtered_ed.empty:
        st.markdown("#### 🤖 CALIBER360 Insight")
        insight = generate_insight(title, filtered_as, filtered_ed)
        st.info(insight)


