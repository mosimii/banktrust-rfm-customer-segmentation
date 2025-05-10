import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#page configuration
st.set_page_config(page_title="BankTrust Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    rfm = pd.read_csv('segmented_data.csv')

    # Merge on CustomerID if present
    if 'CustomerID' in df.columns:
        merged_rfm = pd.merge(rfm, df[['CustomerID']], left_index=True, right_index=True)
    else:
        merged_rfm = rfm  # fallback if no CustomerID

    return df, merged_rfm

df, rfm = load_data()


# --- Sidebar ---
with st.sidebar:
    st.title("🔍 Dashboard Filters")
    st.subheader("🧩 Filter Customers by Segment and Cluster")
    segments = st.multiselect("Select Segment", sorted(rfm['Segments'].unique()), default=sorted(rfm['Segments'].unique()))
    clusters = st.multiselect("Select Cluster", sorted(rfm['Cluster'].unique()), default=sorted(rfm['Cluster'].unique()))

    st.subheader("📈 Chart Display Options")
    # Master toggle
    select_all = st.checkbox("Select / Unselect All Charts", value=True)

    # Individual toggles controlled by master toggle
    show_data_features = st.checkbox("Show Data Features Plots", value=select_all)
    show_cluster_distribution = st.checkbox("Show Cluster Size Bar", value=select_all)
    show_segment_funnel = st.checkbox("Show Segment Value Funnel", value=select_all)
    show_segment_table = st.checkbox("Show Segment Info Table", value=select_all)
    show_lifecycle_pie = st.checkbox("Show Segment Lifecycle Pie", value=select_all)
    show_segment_composition = st.checkbox("Show Segment Composition Table", value=select_all)
    show_cluster_profiles = st.checkbox("Show Cluster Profiles", value=select_all)
    show_retention_strategies = st.checkbox("Show Retention Strategies", value=select_all)

    st.markdown("---")
    st.caption("Optimized for performance • Mobile-compatible")

filtered_df = rfm[(rfm['Segments'].isin(segments)) & (rfm['Cluster'].isin(clusters))]

# --- Dashboard Title ---
st.title("🏦 BankTrust Customer Segmentation Dashboard")

# --- KPIs ---
st.markdown("### 📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Number of Customers", len(filtered_df))
col2.metric("Avg. Recency", f"{filtered_df['Recency'].mean():.1f} days")
col3.metric("Avg. Frequency", f"{filtered_df['Frequency'].mean():.1f} txns")
col4.metric("Avg. Monetary", f"₹{filtered_df['Monetary'].mean():,.0f}")
largest_seg = filtered_df['Segments'].value_counts().idxmax()
col5.metric("Top Segment", largest_seg)



# --- Data Features Plot ---
if show_data_features:
    st.markdown("### 🔍 Data Features Overview")

    sns.set_theme(style="whitegrid")

    # Row 1: Age + Frequency
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🎂 Customer Age Distribution")
        fig_age, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df['Age'], bins=30, kde=True, color='purple', ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig_age)

    with col2:
        st.markdown("##### 🔄 Transaction Frequency Distribution")
        fig_freq, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(rfm['Frequency'], bins=30, kde=True, color='blue', ax=ax)
        ax.set_title("Transaction Frequency Distribution")
        st.pyplot(fig_freq)

    # Row 2: Monetary + Recency
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 💰 Monetary Value Distribution (Log Scale)")
        fig_money, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(rfm['Monetary'], bins=np.logspace(0, np.log10(rfm['Monetary'].max()), 30), color='green', ax=ax)
        ax.set_xscale('log')
        ax.set_title("Monetary Value Distribution (Log Scale)")
        ax.set_xlabel("Monetary (Log Scale)")
        st.pyplot(fig_money)

    with col4:
        st.markdown("##### ⏳ Recency Distribution")
        fig_rec, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(rfm['Recency'], bins=30, kde=True, color='orange', ax=ax)
        ax.set_title("Recency Distribution")
        st.pyplot(fig_rec)


# --- Cluster Size Bar ---
# Define consistent colors for segments
segment_color_map = {
    "Low": "#EF553B",    # red
    "Medium": "#636EFA", # blue
    "High": "#00CC96"    # green
}



# Cluster Size Bar
if show_cluster_distribution:
    st.markdown("### 👥 Cluster Sizes")
    cluster_counts = filtered_df['Cluster'].value_counts().sort_index()
    fig_cluster = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Customers'},
        title="Number of Customers per Cluster",
        color=cluster_counts.index.astype(str)
    )
    fig_cluster.update_layout(height=300)
    st.plotly_chart(fig_cluster, use_container_width=True)



# --- Segment Funnel ---
if show_segment_funnel:
    st.markdown("### 🔄 Segment Value Funnel")
    seg_mon = filtered_df.groupby('Segments')['Monetary'].mean().sort_values(ascending=False).reset_index()
    fig_funnel = px.funnel(
        seg_mon,
        y='Segments',
        x='Monetary',
        color='Segments',
        color_discrete_map=segment_color_map,
        title="Average Monetary Value by Segment"
    )
    fig_funnel.update_layout(height=300)
    st.plotly_chart(fig_funnel, use_container_width=True)



# --- Segment Info Table ---
if show_segment_table:
    st.markdown("### 📘 Segment Description Table")

    segment_info = pd.DataFrame({
        "Segment Name": ["Low", "Medium", "High"],
        "Description": [
            "Low-value customers🪙: infrequent, low spending, long inactive",
            "Medium-value customers🔶: moderate activity and spending",
            "High-value customers💎: frequent, high spenders, recent activity"
        ]
    })

    # RGBA colors for transparency (alpha = 0.2 for light tint)
    row_colors = {
        "Low": "rgba(239, 85, 59, 0.2)",     # red tint
        "Medium": "rgba(99, 110, 250, 0.2)",  # blue tint
        "High": "rgba(0, 204, 150, 0.2)"      # green tint
    }

    # Apply transparent styling
    def highlight_row(row):
        color = row_colors.get(row['Segment Name'], "rgba(255,255,255,0)")
        return [f'background-color: {color};' for _ in row]

    styled_table = segment_info.style.apply(highlight_row, axis=1)

    st.write(styled_table)




# --- Segment Lifecycle Pie ---
if show_lifecycle_pie:
    st.markdown("### 📊 Segment Lifecycle Distribution")
    st.write("This chart shows the percentage of customers in each segment, reflecting the **customer count share**.")

    seg_dist = filtered_df['Segments'].value_counts().reset_index()
    seg_dist.columns = ['Segment', 'Count']

    fig_pie = px.pie(
        seg_dist,
        names='Segment',
        values='Count',
        hole=0.6,
        title=" ",
        color='Segment',
        color_discrete_map=segment_color_map
    )

    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        pull=[0.05] * len(seg_dist),
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )

    fig_pie.update_layout(
        showlegend=True,
        legend_title_text='Segments',
        title_x=0.5,
        height=500,
        font=dict(size=14),
        annotations=[dict(text='Segments', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig_pie, use_container_width=True)




# --- Segment Composition Table ---
if show_segment_composition:
    st.markdown("### 📋 Segment Composition Overview")
    st.write("This chart shows the percentage of **total monetary value** contributed by each segment, reflecting their financial importance.")

    # Aggregate composition data
    comp_df = filtered_df.groupby('Segments').agg({
        'Monetary': 'sum',
        'Recency': 'count'
    }).rename(columns={'Recency': 'Customers', 'Monetary': 'Total Monetary'}).reset_index()

    comp_df['% of Customers'] = (comp_df['Customers'] / comp_df['Customers'].sum() * 100).round(1)
    comp_df['% of Value'] = (comp_df['Total Monetary'] / comp_df['Total Monetary'].sum() * 100).round(1)

    # Define color map
    segment_color_map = {
        "Low": "#EF553B",
        "Medium": "#636EFA",
        "High": "#00CC96"
    }

    # Create bar chart
    fig_comp = px.bar(
        comp_df.sort_values(by='% of Value', ascending=False),
        x='Segments',
        y='% of Value',
        text='% of Value',
        color='Segments',
        color_discrete_map=segment_color_map,
        title="Customer Value Contribution by Segment"
    )

    fig_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

    fig_comp.update_layout(
        yaxis_title="% of Total Value",
        xaxis_title="Segments",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        showlegend=False,
        height=400,
        bargap=0.4
    )

    st.plotly_chart(fig_comp, use_container_width=True)




# --- Cluster Profiles Section ---
if show_cluster_profiles:
    st.write("Cluster and Segment Cross-tab:")
    st.write(pd.crosstab(filtered_df['Cluster'], filtered_df['Segments']))
    
    st.markdown("### 🏆 Cluster Profiles")

    for cluster_id in sorted(filtered_df['Cluster'].unique()):
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster_id]

        with st.expander(f"Cluster {cluster_id}", expanded=False):
            # Corrected cluster descriptions
            if cluster_id == 2:
                st.write("💎 High-value customers: frequent, high-spending, recent activity.")
            elif cluster_id == 0:
                st.write("📦 Medium-value customers: moderate spending, moderate activity.")
            elif cluster_id == 1:
                st.write("📉 Low-value customers: infrequent, low-spending, at risk or inactive.")
            else:
                st.write("📦 Other customer group.")

            # Segment summary inside this cluster
            segment_counts = cluster_data['Segments'].value_counts()
            segment_summary = ", ".join([f"{seg}: {count}" for seg, count in segment_counts.items()])
            st.write(f"**Segment composition:** {segment_summary}")

            # Select top 3 customers per segment (only from existing segments)
            top_customers_list = []
            for segment in segment_counts.index:
                seg_data = cluster_data[cluster_data['Segments'] == segment]
                top_seg_customers = seg_data.sort_values(by='Monetary', ascending=False).head(3)
                top_customers_list.append(top_seg_customers)

            # Combine and clean top customers
            if top_customers_list:
                top_customers = pd.concat(top_customers_list).reset_index(drop=True)

                # Remove unwanted columns
                cols_to_drop = [col for col in ['Weighted Score', 'Weighted Segments'] if col in top_customers.columns]
                top_customers = top_customers.drop(columns=cols_to_drop)

                # Rename and format
                top_customers.rename(columns={
                    'Recency': 'Recency (days)',
                    'Monetary': 'Monetary Value',
                    'Segments': 'Segment'
                }, inplace=True)

                if 'Monetary Value' in top_customers.columns:
                    top_customers['Monetary Value'] = top_customers['Monetary Value'].apply(lambda x: f"{x:,.2f}")

                # Display compact table
                st.dataframe(top_customers, use_container_width=True)
            else:
                st.write("No customers available in this cluster.")



# --- Strategies Table ---
if show_retention_strategies:
    st.markdown("### 💡 Recommended Retention Strategies by Customer Group")

    strategies = {
        "High-Value Customers": {
            "emoji": "💎",
            "strategy": (
                "**Focus on retention**:\n"
                "- Offer loyalty rewards and VIP programs.\n"
                "- Provide exclusive deals and early access.\n"
                "- Maintain high-touch communication to keep them engaged."
            )
        },
        "Medium-Value Customers": {
            "emoji": "🔶",
            "strategy": (
                "**Boost engagement**:\n"
                "- Send targeted promotions or seasonal offers.\n"
                "- Provide incentives to increase frequency.\n"
                "- Highlight upsell or cross-sell opportunities."
            )
        },
        "Low-Value Customers": {
            "emoji": "🪙",
            "strategy": (
                "**Re-engage or win back**:\n"
                "- Use special discounts or win-back campaigns.\n"
                "- Personalize outreach to understand drop-off reasons.\n"
                "- Simplify reactivation steps and offer clear value."
            )
        }
    }

    for group, details in strategies.items():
        with st.expander(f"{details['emoji']} {group}"):
            st.markdown(details['strategy'])


# --- Download Filtered Data ---
# st.markdown("### 💾 Download Filtered Data")
# csv = filtered_df.to_csv(index=False)
# st.download_button("📥 Download CSV", data=csv, file_name="filtered_data.csv")