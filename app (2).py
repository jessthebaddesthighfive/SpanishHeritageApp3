import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime
from data_sources import HistoricalDataSource
from regression_analysis import RegressionAnalyzer

# Set page configuration
st.set_page_config(
    page_title="Latin American Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header with user name
st.title("📊 Comprehensive Latin American Historical Data Analysis")

# User name input
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

user_name = st.text_input("Enter your name:", value=st.session_state.user_name, placeholder="Your Name")
if user_name:
    st.session_state.user_name = user_name
    st.markdown(f"### *Created by: {user_name}*")
else:
    st.markdown("### *Created by: [Enter your name above]*")
st.markdown("---")

# Initialize data source
@st.cache_data
def load_data_source():
    return HistoricalDataSource()

data_source = load_data_source()

# Sidebar configuration
st.sidebar.title("Analysis Configuration")

# Data category selection
categories = [
    "Population",
    "Unemployment Rate", 
    "Education Levels",
    "Life Expectancy",
    "Average Wealth",
    "Average Income",
    "Birth Rate",
    "Immigration out of Country",
    "Murder Rate"
]

selected_category = st.sidebar.selectbox("Select Data Category:", categories)

# Country selection
countries = ['Mexico', 'Brazil', 'Argentina']
selected_countries = st.sidebar.multiselect(
    "Select Countries:", 
    countries, 
    default=['Mexico']
)

# Time increment selection
time_increment = st.sidebar.slider("Time Increment (years):", 1, 10, 5)

# Regression degree selection
regression_degree = st.sidebar.slider("Polynomial Degree:", 3, 8, 3)

# Analysis options
st.sidebar.subheader("Analysis Options")
show_comparison = st.sidebar.checkbox("Multi-Country Comparison")
show_us_latin = st.sidebar.checkbox("Include US Latin Groups")
extrapolate_years = st.sidebar.slider("Extrapolate Years into Future:", 0, 30, 10)

# Function to get data based on category
@st.cache_data
def get_category_data(category):
    if category == "Population":
        return data_source.get_population_data()
    elif category == "Unemployment Rate":
        return data_source.get_unemployment_data()
    elif category == "Education Levels":
        return data_source.get_education_data()
    elif category == "Life Expectancy":
        return data_source.get_life_expectancy_data()
    elif category == "Average Wealth":
        return data_source.get_wealth_data()
    elif category == "Average Income":
        return data_source.get_income_data()
    elif category == "Birth Rate":
        return data_source.get_birth_rate_data()
    elif category == "Immigration out of Country":
        return data_source.get_immigration_data()
    elif category == "Murder Rate":
        return data_source.get_murder_rate_data()

# Load data
if selected_countries:
    df = get_category_data(selected_category)
    
    # Apply time increment filter
    if df is not None:
        if time_increment > 1:
            df_filtered = df[df['Year'] % time_increment == 0]
        else:
            df_filtered = df.copy()
    else:
        st.error("Unable to load data for the selected category.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {selected_category} Analysis")
        
        # Data table section
        st.subheader("📋 Raw Historical Data")
        
        # Create editable data table
        edited_df = st.data_editor(
            df_filtered, 
            height=400,
            use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", disabled=True),
            }
        )
        
        st.caption(f"Data source: World Bank, UN Statistics, National Statistical Offices. Time increment: {time_increment} year(s)")
    
    with col2:
        st.subheader("🎛️ Regression Controls")
        
        # Specific year prediction
        st.subheader("🔮 Specific Year Prediction")
        prediction_year = st.number_input(
            "Enter year for prediction:",
            min_value=1900,
            max_value=2100,
            value=2030
        )
        
        # Rate of change calculator
        st.subheader("📈 Rate of Change Calculator")
        year1 = st.number_input("Start Year:", min_value=1950, max_value=2023, value=2000)
        year2 = st.number_input("End Year:", min_value=1951, max_value=2024, value=2020)
    
    # Analysis for each selected country
    for country in selected_countries:
        if country in edited_df.columns:
            st.markdown("---")
            st.subheader(f"🌎 Analysis for {country}")
            
            # Get data for analysis
            x_data = edited_df['Year'].to_numpy()
            y_data = edited_df[country].to_numpy()
            
            # Remove any NaN values
            mask = ~np.isnan(y_data)
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            if len(x_data) < 4:
                st.warning(f"Insufficient data points for {country}. Need at least 4 points for analysis.")
                continue
            
            # Fit regression model
            analyzer = RegressionAnalyzer(degree=regression_degree)
            analyzer.fit_model(x_data, y_data)
            
            # Create three columns for layout
            analysis_col1, analysis_col2, analysis_col3 = st.columns([2, 1, 1])
            
            with analysis_col1:
                # Create visualization
                comparison_data = []
                if show_comparison:
                    for other_country in selected_countries:
                        if other_country != country and other_country in edited_df.columns:
                            other_y = edited_df[other_country].to_numpy()
                            other_mask = ~np.isnan(other_y)
                            other_x = edited_df['Year'].to_numpy()[other_mask]
                            other_y = other_y[other_mask]
                            comparison_data.append((other_country, other_x, other_y))
                
                fig = analyzer.create_visualization(
                    x_data, y_data, country, selected_category,
                    extrapolate_years, comparison_data if show_comparison else None
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with analysis_col2:
                # Model equation and statistics
                st.subheader("📊 Model Information")
                st.write("**Equation:**")
                st.code(analyzer.get_equation_string())
                st.write(f"**R² Score:** {analyzer.r2_score:.4f}")
                st.write(f"**Degree:** {regression_degree}")
                
                # Specific predictions
                if prediction_year:
                    pred_value = analyzer.predict([prediction_year])[0]
                    st.write("**Prediction:**")
                    st.info(f"Year {prediction_year}: {pred_value:.2f}")
                
                # Rate of change
                if year1 < year2:
                    rate = analyzer.calculate_rate_of_change(year1, year2)
                    st.write("**Rate of Change:**")
                    st.info(f"{year1}-{year2}: {rate:.4f} units/year")
            
            with analysis_col3:
                # Function analysis summary
                analysis = analyzer.analyze_function_behavior(
                    x_data, y_data, selected_category, country
                )
                
                st.subheader("🔍 Function Analysis")
                st.write(f"**Domain:** {analysis['domain']}")
                st.write(f"**Range:** {analysis['range']}")
                
                if analysis['critical_points']:
                    st.write("**Critical Points:**")
                    for i, (x, y) in enumerate(analysis['critical_points']):
                        st.write(f"• Year {int(x)}: {y:.2f}")
                
                if analysis['inflection_points']:
                    st.write("**Inflection Points:**")
                    for i, (x, y) in enumerate(analysis['inflection_points']):
                        st.write(f"• Year {int(x)}: {y:.2f}")
            
            # Detailed interpretation
            st.subheader("🎯 Real-World Analysis & Historical Context")
            
            analysis = analyzer.analyze_function_behavior(x_data, y_data, selected_category, country)
            
            for interpretation in analysis['interpretation']:
                st.write(f"• {interpretation}")
            
            # Historical context and conjectures
            st.subheader("🏛️ Historical Context & Conjectures")
            
            # Add specific historical context based on country and time period
            if country == "Mexico":
                if selected_category in ["Population", "Immigration out of Country"]:
                    st.write("• **1994 NAFTA Implementation**: Significant economic changes affecting migration patterns and economic indicators.")
                    st.write("• **2008 Financial Crisis**: Global economic downturn impacted employment and migration trends.")
                    st.write("• **Drug War Period (2006-2012)**: Security issues influenced various social and economic metrics.")
                elif selected_category == "Murder Rate":
                    st.write("• **Drug War Escalation (2006-2012)**: Sharp increase in violence during Felipe Calderón's administration.")
                    st.write("• **Security Strategy Changes**: Variations in murder rates reflect different approaches to organized crime.")
            
            elif country == "Brazil":
                if selected_category in ["Economic", "Average Wealth", "Average Income"]:
                    st.write("• **Real Plan (1994)**: Currency stabilization significantly impacted economic indicators.")
                    st.write("• **Commodities Boom (2003-2011)**: High demand for Brazilian exports drove economic growth.")
                    st.write("• **Political Crisis (2014-2016)**: Economic recession and political instability affected various metrics.")
            
            elif country == "Argentina":
                if selected_category in ["Average Wealth", "Average Income", "Unemployment Rate"]:
                    st.write("• **Economic Crisis (2001-2002)**: Severe economic collapse with lasting effects on multiple indicators.")
                    st.write("• **Commodity Price Boom**: Agriculture-dependent economy benefited from global price increases.")
                    st.write("• **Currency Controls**: Various exchange rate policies influenced economic metrics.")
            
            # Future extrapolation analysis
            if extrapolate_years > 0:
                st.subheader("🔮 Future Projections & Extrapolation Analysis")
                
                future_year = max(x_data) + extrapolate_years
                future_value = analyzer.predict([future_year])[0]
                
                st.write(f"**Projection for {int(future_year)}:** {future_value:.2f}")
                
                # Generate contextual prediction
                if selected_category == "Population":
                    st.write(f"According to the regression model, the population of {country} is projected to reach approximately {future_value:.1f} million by {int(future_year)}.")
                elif selected_category == "Life Expectancy":
                    st.write(f"The model projects life expectancy in {country} to reach {future_value:.1f} years by {int(future_year)}.")
                elif selected_category == "Unemployment Rate":
                    st.write(f"Based on historical trends, unemployment in {country} is projected to be {future_value:.1f}% in {int(future_year)}.")
                
                # Add caveats about extrapolation
                st.warning("⚠️ **Extrapolation Limitations**: These projections are based on historical patterns and may not account for future policy changes, economic shocks, or demographic transitions.")
    
    # US Latin Groups Comparison
    if show_us_latin:
        st.markdown("---")
        st.subheader("🇺🇸 US Latin American Population Groups Analysis")
        
        # Map category to available US data
        us_category = selected_category
        if selected_category not in ["Population", "Average Income", "Education Levels"]:
            us_category = "Education Levels"  # Default fallback
        
        us_df = data_source.get_us_latin_data(us_category)
        
        st.subheader(f"📊 {us_category} Data for US Latin Groups")
        st.dataframe(us_df, use_container_width=True)
        
        # Create comparison visualization
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, group in enumerate(us_df.columns[1:]):  # Skip 'Year' column
            fig.add_trace(go.Scatter(
                x=us_df['Year'],
                y=us_df[group],
                mode='lines+markers',
                name=group,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'{us_category} Trends for US Latin American Groups',
            xaxis_title='Year',
            yaxis_title=us_category,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis for US groups
        st.write("**Key Observations:**")
        if us_category == "Population":
            st.write("• Mexican Americans represent the largest Latin American group in the US with consistent growth.")
            st.write("• Cuban Americans show more modest growth, reflecting migration patterns and policies.")
            st.write("• Salvadoran Americans show significant growth, particularly after 1990s migration waves.")
        elif us_category == "Average Income":
            st.write("• Cuban Americans typically show higher average incomes, reflecting education levels and settlement patterns.")
            st.write("• Income disparities reflect different migration histories and integration experiences.")
        else:  # Education Levels
            st.write("• Education levels have generally increased across all groups over time.")
            st.write("• Different groups show varying rates of educational attainment improvement.")
    
    # Printer-friendly export
    st.markdown("---")
    st.subheader("🖨️ Export Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Generate Summary Report"):
            # Create a comprehensive text report
            report_lines = [
                f"LATIN AMERICAN DATA ANALYSIS REPORT",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"",
                f"ANALYSIS CONFIGURATION:",
                f"Category: {selected_category}",
                f"Countries: {', '.join(selected_countries)}",
                f"Time Increment: {time_increment} years",
                f"Polynomial Degree: {regression_degree}",
                f"",
                f"DATA SUMMARY:",
            ]
            
            for country in selected_countries:
                if country in edited_df.columns:
                    x_data = edited_df['Year'].to_numpy()
                    y_data = edited_df[country].to_numpy()
                    mask = ~np.isnan(y_data)
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                    
                    if len(x_data) >= 4:
                        analyzer = RegressionAnalyzer(degree=regression_degree)
                        analyzer.fit_model(x_data, y_data)
                        
                        report_lines.extend([
                            f"",
                            f"{country.upper()} ANALYSIS:",
                            f"Equation: {analyzer.get_equation_string()}",
                            f"R² Score: {analyzer.r2_score:.4f}",
                            f"Data Range: {min(y_data):.2f} to {max(y_data):.2f}",
                            f"Time Period: {int(min(x_data))} to {int(max(x_data))}",
                        ])
                        
                        if extrapolate_years > 0:
                            future_year = max(x_data) + extrapolate_years
                            future_value = analyzer.predict([future_year])[0]
                            report_lines.append(f"Projection for {int(future_year)}: {future_value:.2f}")
            
            # Display report in text area
            report_text = "\n".join(report_lines)
            st.text_area("Summary Report:", report_text, height=400)
    
    with col2:
        if st.button("📊 Export Data as CSV"):
            # Create downloadable CSV
            csv_buffer = io.StringIO()
            edited_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{selected_category}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Read the current app.py file
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        st.download_button(
            label="📱 Download app.py",
            data=app_content,
            file_name="app.py",
            mime="text/plain",
            help="Download the complete app.py file for GitHub deployment"
        )
    
    with col4:
        # Read the requirements file
        with open('app_requirements.txt', 'r', encoding='utf-8') as f:
            req_content = f.read()
        
        st.download_button(
            label="📦 Download requirements.txt",
            data=req_content,
            file_name="requirements.txt",
            mime="text/plain",
            help="Download requirements.txt for Streamlit deployment"
        )
    
    # Usage tips moved below
    st.info("📋 **Usage Tips:**\n• Adjust time increments for smoother curves\n• Use higher polynomial degrees for complex patterns\n• Enable comparisons to see relative trends\n• Extrapolation shows potential future scenarios\n• Download the app.py and requirements.txt files above for GitHub deployment")

else:
    st.warning("Please select at least one country to begin analysis.")
    
    # Show sample data structure when no countries selected
    st.subheader("📋 Available Data Structure")
    sample_df = get_category_data("Population")
    if sample_df is not None:
        st.dataframe(sample_df.head())
    else:
        st.error("Unable to load sample data.")
    
    st.subheader("ℹ️ Application Features")
    st.write("""
    **This application provides:**
    
    • **Historical Data Analysis**: Real data from World Bank, UN Statistics, and national sources
    • **Polynomial Regression**: Advanced curve fitting with degrees 3 and higher  
    • **Function Analysis**: Critical points, inflection points, and rate calculations
    • **Multi-Country Comparisons**: Side-by-side analysis of trends
    • **US Latin Groups**: Comparative analysis of demographic groups in the United States
    • **Future Projections**: Extrapolation capabilities with uncertainty indicators
    • **Interactive Tools**: Editable data tables and customizable visualizations
    • **Export Options**: Printer-friendly reports and CSV downloads
    
    **Supported Countries:** Mexico, Brazil, Argentina (3 wealthiest Latin American nations)
    
    **Data Categories:** Population, Unemployment, Education, Life Expectancy, Wealth, Income, Birth Rate, Immigration, Murder Rate
    
    **Time Coverage:** Up to 70+ years of historical data depending on category
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>📊 Latin American Historical Data Analysis Tool | Built with Streamlit</p>
    <p>Data sources: World Bank, UN Statistics Division, National Statistical Offices</p>
    </div>
    """, 
    unsafe_allow_html=True
)
