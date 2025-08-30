import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# =====================
# Load Data
# =====================
file_path = "financial_data.xlsx"  # <-- replace with your actual Excel file

try:
    df = pd.read_excel(file_path)
    st.sidebar.write("✅ File loaded successfully")
    
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Check if required columns exist
required_columns = ["Date type", "Date", "Segment", "Category", "Amount"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Clean and standardize data
# Convert Amount to numeric, handling comma-separated values and hash symbols
df["Amount"] = df["Amount"].astype(str).str.replace(',', '').str.replace('#', '0')
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

# Clean Date column
df["Date"] = df["Date"].astype(str).str.strip()

if df.empty:
    st.error("No valid data found after processing.")
    st.stop()

# =====================
# Sidebar Filters
# =====================
st.sidebar.header("Filters")

# 1) Period Type Selector
period_type_options = sorted(df["Date type"].unique())
selected_period_type = st.sidebar.selectbox("Period Type", period_type_options)

# 2) Filter data based on selected period type
period_filtered_df = df[df["Date type"] == selected_period_type].copy()

# 3) Period Selector (Single dropdown based on period type)
period_options = sorted(period_filtered_df["Date"].unique())
selected_period = st.sidebar.selectbox(f"Select {selected_period_type}", period_options)

# 4) Segment Filter (Single dropdown)
segments = ["All"] + sorted([seg for seg in df["Segment"].dropna().unique() if seg != "All"])
selected_segment = st.sidebar.selectbox("Segment", segments)


# =====================
# Filter Data
# =====================
filtered_df = df[
    (df["Date type"] == selected_period_type) & 
    (df["Date"] == selected_period)
].copy()

if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["Segment"] == selected_segment]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =====================
# Calculations
# =====================
def calc_financial_metrics(data):
    """Calculate financial metrics from category-based data"""
    
    # Group by category and sum amounts
    category_totals = data.groupby("Category")["Amount"].sum()
    
    # Get values for each category (default to 0 if not present)
    revenue = category_totals.get("Revenue", 0)
    direct_exp = category_totals.get("Direct Expense", 0)
    indirect_exp = category_totals.get("Indirect Expense", 0)
    interest = category_totals.get("Interest", 0)
    
    # Calculate derived metrics
    gross_profit = revenue - direct_exp
    ebitda = gross_profit - indirect_exp
    net_profit = ebitda - interest
    
    return revenue, direct_exp, indirect_exp, interest, gross_profit, ebitda, net_profit

revenue, direct_exp, indirect_exp, interest, gross_profit, ebitda, net_profit = calc_financial_metrics(filtered_df)

# =====================
# Main Dashboard
# =====================
st.title("📊 Financial Dashboard")

# Show current selection
st.info(f"**Period Type:** {selected_period_type} | **Period:** {selected_period} | **Segment:** {selected_segment}")

# =====================
# KPI Cards - Three Rows
# =====================
st.subheader("💰 Financial Metrics")

# Row 1: Revenue
col1 = st.columns(1)[0]
with col1:
    st.metric("Revenue", f"₹{revenue:,.0f}")

# Row 2: Expenses
col2, col3, col4 = st.columns(3)
with col2:
    st.metric("Direct Expense", f"₹{direct_exp:,.0f}")
with col3:
    st.metric("Indirect Expense", f"₹{indirect_exp:,.0f}")
with col4:
    st.metric("Interest", f"₹{interest:,.0f}")

# Row 3: Profits
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Gross Profit", f"₹{gross_profit:,.0f}")
with col6:
    st.metric("EBITDA", f"₹{ebitda:,.0f}")
with col7:
    st.metric("Net Profit", f"₹{net_profit:,.0f}")

# =====================
# Line Chart: Revenue vs Direct Expense
# =====================
st.subheader("📈 Revenue vs Direct Expense Trend")

# Get data for line chart (all months regardless of current filter)
if selected_segment == "All":
    line_df = df[df["Date type"] == "Month"].copy()
else:
    line_df = df[(df["Date type"] == "Month") & (df["Segment"] == selected_segment)].copy()

# Group by month and category, then pivot
line_data = line_df.groupby(["Date", "Category"])["Amount"].sum().unstack(fill_value=0).reset_index()

# Detect possible date/month column
date_col = None
for col in ["Month", "Date", "Period", "Month-Year"]:
    if col in line_data.columns:
        date_col = col
        break

if date_col is None:
    st.warning("No date/month column found in data for line chart.")
else:
    # Convert to datetime
    line_data["Month_dt"] = pd.to_datetime(line_data[date_col], errors="coerce")

    # Create month labels (e.g., Apr, May, Jun)
    line_data["Month_Label"] = line_data["Month_dt"].dt.strftime("%b")

    # Get Revenue and Direct Expense columns if they exist
    revenue_col = "Revenue" if "Revenue" in line_data.columns else None
    direct_exp_col = "Direct Expense" if "Direct Expense" in line_data.columns else None

    if revenue_col or direct_exp_col:
        # Filter out months where both Revenue and Direct Expense are 0
        if revenue_col and direct_exp_col:
            line_data = line_data[(line_data[revenue_col] != 0) | (line_data[direct_exp_col] != 0)]
            cols_to_plot = [revenue_col, direct_exp_col]
        elif revenue_col:
            line_data = line_data[line_data[revenue_col] != 0]
            cols_to_plot = [revenue_col]
        else:
            line_data = line_data[line_data[direct_exp_col] != 0]
            cols_to_plot = [direct_exp_col]

        if not line_data.empty:
            fig1 = px.line(
                line_data,
                x="Month_Label",
                y=cols_to_plot,
                markers=True,
                title="Revenue vs Direct Expense (Monthly)"
            )

            # Add data labels
            for i, trace in enumerate(fig1.data):
                trace.text = [f"₹{val:,.0f}" for val in trace.y]
                trace.textposition = "top center"
                trace.mode = "lines+markers+text"

            fig1.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("No data available for line chart (all months have zero revenue and direct expense).")
    else:
        st.warning("Revenue or Direct Expense data not found for line chart.")


# =====================
# Segment Wise Gross Profit Bar Chart
# =====================
st.subheader("🎯 Segment Wise Gross Profit")

# Get segment data for current period - filter out "All" scenario
segment_df = df[
    (df["Date type"] == selected_period_type) & 
    (df["Date"] == selected_period)
].copy()

# Remove any rows where segment might be "All" or empty
segment_df = segment_df[segment_df["Segment"].notna() & (segment_df["Segment"] != "All")]

# Group by segment and category
seg_data = segment_df.groupby(["Segment", "Category"])["Amount"].sum().unstack(fill_value=0).reset_index()

# Calculate Gross Profit if Revenue and Direct Expense exist
if "Revenue" in seg_data.columns and "Direct Expense" in seg_data.columns:
    seg_data["Gross Profit"] = seg_data["Revenue"] - seg_data["Direct Expense"]
    
    # Select columns for chart
    chart_columns = []
    if "Revenue" in seg_data.columns:
        chart_columns.append("Revenue")
    if "Direct Expense" in seg_data.columns:
        chart_columns.append("Direct Expense")
    if "Gross Profit" in seg_data.columns:
        chart_columns.append("Gross Profit")
    
    if chart_columns and not seg_data.empty:
        fig2 = px.bar(seg_data, x="Segment", y=chart_columns, barmode="group",
                      title="Segment Wise Gross Profit")
        
        # Add data labels to bar chart
        for trace in fig2.data:
            trace.text = [f"₹{val:,.0f}" if val != 0 else "" for val in trace.y]
            trace.textposition = "outside"
            trace.texttemplate = "%{text}"
        
        fig2.update_layout(yaxis_title="Amount (₹)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No segment data available for chart.")
else:
    st.warning("Revenue or Direct Expense data not found for segment analysis.")

# =====================
# P&L Summary Table (Transposed)
# =====================
st.subheader("📑 P&L Summary")

# Create P&L for all months (or selected period type)
if selected_segment == "All":
    pnl_df = df[df["Date type"] == "Month"].copy()
else:
    pnl_df = df[(df["Date type"] == "Month") & (df["Segment"] == selected_segment)].copy()

# Group by month and category
pnl_pivot = pnl_df.groupby(["Date", "Category"])["Amount"].sum().unstack(fill_value=0).reset_index()

# Rename Date to Period for consistency
pnl_pivot = pnl_pivot.rename(columns={"Date": "Period"})

# Add financial calculation columns
if "Revenue" in pnl_pivot.columns:
    pnl_pivot["Revenue"] = pnl_pivot["Revenue"]
else:
    pnl_pivot["Revenue"] = 0

if "Direct Expense" in pnl_pivot.columns:
    pnl_pivot["Direct Expense"] = pnl_pivot["Direct Expense"]
else:
    pnl_pivot["Direct Expense"] = 0

if "Indirect Expense" in pnl_pivot.columns:
    pnl_pivot["Indirect Expense"] = pnl_pivot["Indirect Expense"]
else:
    pnl_pivot["Indirect Expense"] = 0

if "Interest" in pnl_pivot.columns:
    pnl_pivot["Interest"] = pnl_pivot["Interest"]
else:
    pnl_pivot["Interest"] = 0

# Calculate derived metrics
pnl_pivot["Gross Profit"] = pnl_pivot["Revenue"] - pnl_pivot["Direct Expense"]
pnl_pivot["Gross Margin"] = pnl_pivot["Gross Profit"] / pnl_pivot["Revenue"].replace(0, 1)

pnl_pivot["EBITDA"] = pnl_pivot["Gross Profit"] - pnl_pivot["Indirect Expense"]
pnl_pivot["EBITDA Margin"] = pnl_pivot["EBITDA"] / pnl_pivot["Revenue"].replace(0, 1)

pnl_pivot["Net Profit"] = pnl_pivot["EBITDA"] - pnl_pivot["Interest"]
pnl_pivot["Net Margin"] = pnl_pivot["Net Profit"] / pnl_pivot["Revenue"].replace(0, 1)

# Filter out months where both Revenue and Direct Expense are 0
pnl_summary = pnl_pivot[(pnl_pivot["Revenue"] != 0) | (pnl_pivot["Direct Expense"] != 0)].copy()

# Select only the required columns for P&L
pnl_columns = ["Period", "Revenue", "Direct Expense", "Gross Profit", "Gross Margin", 
               "Indirect Expense", "EBITDA", "EBITDA Margin", "Interest", "Net Profit", "Net Margin"]

pnl_final = pnl_summary[pnl_columns].copy()

if not pnl_final.empty:
    # Add Total Column
    total_revenue = pnl_final["Revenue"].sum()
    total_row = {
        "Period": "Total",
        "Revenue": total_revenue,
        "Direct Expense": pnl_final["Direct Expense"].sum(),
        "Gross Profit": pnl_final["Gross Profit"].sum(),
        "Gross Margin": pnl_final["Gross Profit"].sum() / total_revenue if total_revenue != 0 else 0,
        "Indirect Expense": pnl_final["Indirect Expense"].sum(),
        "EBITDA": pnl_final["EBITDA"].sum(),
        "EBITDA Margin": pnl_final["EBITDA"].sum() / total_revenue if total_revenue != 0 else 0,
        "Interest": pnl_final["Interest"].sum(),
        "Net Profit": pnl_final["Net Profit"].sum(),
        "Net Margin": pnl_final["Net Profit"].sum() / total_revenue if total_revenue != 0 else 0,
    }
    pnl_final = pd.concat([pnl_final, pd.DataFrame([total_row])], ignore_index=True)

    # Transpose the table (Periods as columns, metrics as rows)
    pnl_transposed = pnl_final.set_index("Period").T
    
    # Create HTML styled table with bold headers and proper formatting
    def create_styled_pnl_table(df):
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        
        # Header row with month names (bold)
        html_table += '<tr style="background-color: #f0f2f6;">'
        html_table += '<th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Metrics</th>'
        for col in df.columns:
            html_table += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold;">{col}</th>'
        html_table += '</tr>'
        
        # Data rows
        for idx in df.index:
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{idx}</td>'
            for col in df.columns:
                value = df.loc[idx, col]
                if "Margin" in str(idx):
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"₹{value:,.0f}"
                html_table += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{formatted_value}</td>'
            html_table += '</tr>'
        
        html_table += '</table>'
        return html_table
    
    # Display the styled table
    st.markdown(create_styled_pnl_table(pnl_transposed), unsafe_allow_html=True)

else:
    st.warning("No data available for P&L summary.")

# =====================
# Metrics Table (from Metrics_Calculations)
# =====================
st.subheader("📐 Key Metrics")

try:
    metrics_df = pd.read_excel(file_path, sheet_name="Metrics_Calculations")
except Exception as e:
    st.warning(f"Could not load Metrics_Calculations sheet: {e}")
    metrics_df = pd.DataFrame()

if not metrics_df.empty:
    metrics_df.columns = metrics_df.columns.str.strip()

    # --- Clean numeric columns ---
    def parse_metric(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):   # handle percentages
                try:
                    return float(x.strip("%")) / 100
                except:
                    return np.nan
            x = x.replace(",", "")  # remove commas
            if x.startswith("="):   # skip formulas like "=91.57/1282"
                return np.nan
            try:
                return float(x)
            except:
                return np.nan
        return x

    for col in metrics_df.columns[2:]:  # skip Metric & Formula
        metrics_df[col] = metrics_df[col].apply(parse_metric)

    # --- Quarter mapping ---
    q_map = {
        "Q1": ["April", "May", "June"],
        "Q2": ["July", "August", "September"],
        "Q3": ["October", "November", "December"],
        "Q4": ["January", "February", "March"]
    }

    # --- Convert selected_period to Month Name if needed ---
    selected_month_name = None
    if selected_period_type == "Month":
        try:
            selected_month_name = pd.to_datetime(str(selected_period)).strftime("%B")
        except:
            selected_month_name = str(selected_period)

    # --- Pick the right data ---
    if selected_period_type == "Month":
        if selected_month_name in metrics_df.columns:
            metrics_values = metrics_df[["Metric", selected_month_name]].rename(
                columns={selected_month_name: "Value"}
            )
        else:
            # Month not found → return 0 instead of empty
            metrics_values = metrics_df[["Metric"]].copy()
            metrics_values["Value"] = 0

    elif selected_period_type == "Quarter":
        months = q_map.get(selected_period, [])
        if months:
            available_months = [m for m in months if m in metrics_df.columns]
            metrics_values = metrics_df[["Metric"]].copy()
            if available_months:
                metrics_values["Value"] = metrics_df[available_months].mean(axis=1, skipna=True)
            else:
                metrics_values["Value"] = 0
        else:
            metrics_values = pd.DataFrame(columns=["Metric", "Value"])

    elif selected_period_type == "Year":
        if "Average" in metrics_df.columns:
            metrics_values = metrics_df[["Metric", "Average"]].rename(
                columns={"Average": "Value"}
            )
        else:
            available_months = [col for col in metrics_df.columns if col not in ["Metric", "Formula in Excel/Sheets"]]
            metrics_values = metrics_df[["Metric"]].copy()
            if available_months:
                metrics_values["Value"] = metrics_df[available_months].mean(axis=1, skipna=True)
            else:
                metrics_values["Value"] = 0
    else:
        metrics_values = pd.DataFrame(columns=["Metric", "Value"])

    # --- Formatting ---
    percent_metrics = [
        "Revenue Growth", "EBITDA Margin", "Churn Rate", "DAU/MAU",
        "Mktg Spend/Revenue", "Conversion Rate", "CTR"
    ]
    currency_metrics = ["CAC", "CLTV", "ARPU", "CPL"]

    def format_value(metric, val):
        if pd.isna(val) or val == "":
            return "-"
        try:
            val = float(val)
        except:
            return str(val)

        if metric in percent_metrics:
            return f"{val:.2%}"
        elif metric in currency_metrics:
            return f"₹{val:,.0f}"
        else:
            return f"{val:,.2f}"

    metrics_values["Value"] = metrics_values.apply(
        lambda row: format_value(row["Metric"], row["Value"]), axis=1
    )
    # Reusable function for styled 2-column Metrics table
    def create_styled_metrics_table(df):
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        
        # Header row
        html_table += '<tr style="background-color: #f0f2f6;">'
        html_table += '<th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Metric</th>'
        html_table += '<th style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold;">Value</th>'
        html_table += '</tr>'
        
        # Data rows
        for _, row in df.iterrows():
            metric = row["Metric"]
            value = row["Value"]
            html_table += "<tr>"
            html_table += f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{metric}</td>'
            html_table += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{value}</td>'
            html_table += "</tr>"
        
        html_table += "</table>"
        return html_table


    # --- Show Table ---
    if not metrics_values.empty:
        st.markdown(create_styled_metrics_table(metrics_values), unsafe_allow_html=True)
    else:
        st.info("No metrics available for this selection.")


# =====================
# Data Preview (for debugging)
# =====================
if st.sidebar.checkbox("Show Sample Data"):
    st.subheader("🔍 Sample Data")
    st.dataframe(filtered_df.head(10))
    st.write(f"**Total Rows:** {len(filtered_df)}")
