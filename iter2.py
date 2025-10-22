import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from deap import base, creator, tools, algorithms
import random
import io
import base64

# Page configuration with enhanced styling
st.set_page_config(
    page_title="BytEdge Predictive Maintenance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AI-Themed CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        overflow: auto;
        height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .ai-agent-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
    }
    
    .metric-card h4 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #667eea;
        border-bottom: 2px solid #764ba2;
        padding-bottom: 0.5rem;
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        line-height: 1.6;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        color: white;
        border-left: 4px solid #667eea;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(118, 75, 162, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%);
        color: white;
        border-left: 4px solid #764ba2;
        margin-right: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        width: 100%;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #a0a0a0;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    .risk-high { 
        color: #ff6b6b !important; 
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    .risk-medium { 
        color: #ffd93d !important; 
        font-weight: 800;
    }
    
    .risk-low { 
        color: #6bff6b !important; 
        font-weight: 800;
    }
    
    .live-feed-indicator {
        animation: pulse 2s infinite;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        width: 12px;
        height: 12px;
        display: inline-block;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .vehicle-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .vehicle-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class PredictiveMaintenanceEngine:
    def __init__(self):
        self.material_constants = {
            'fatigue_k': 1.2e-8,
            'wear_k': 3.5e-6,
            'drag_coefficient': 0.65,
            'frontal_area': 8.5,
            'air_density': 1.225,
            'brake_pad_thickness_new': 12.0
        }
    
    def calculate_fatigue_index(self, strain, engine_load):
        return self.material_constants['fatigue_k'] * (strain ** 2) * engine_load
    
    def calculate_drag_force(self, speed_kmh):
        speed_ms = speed_kmh / 3.6
        return 0.5 * self.material_constants['air_density'] * \
               self.material_constants['drag_coefficient'] * \
               self.material_constants['frontal_area'] * \
               (speed_ms ** 2)
    
    def calculate_fuel_rate(self, speed_kmh, engine_load, payload_kg, wind_speed):
        base_rate = 8.0
        speed_effect = 0.02 * (speed_kmh ** 2) * self.material_constants['drag_coefficient']
        load_effect = 0.001 * engine_load * payload_kg / 1000
        wind_effect = 0.1 * abs(wind_speed)
        return base_rate + speed_effect + load_effect + wind_effect + np.random.normal(0, 0.5)
    
    def calculate_brake_wear(self, brake_pressure, speed_kmh, brake_temp):
        braking_work = brake_pressure * (speed_kmh / 3.6) * 0.01
        temp_factor = 1 + 0.002 * max(0, brake_temp - 100)
        return self.material_constants['wear_k'] * braking_work * temp_factor
    
    def calculate_rul(self, current_wear, wear_rate):
        remaining_material = self.material_constants['brake_pad_thickness_new'] - current_wear
        if wear_rate <= 0:
            return float('inf')
        return remaining_material / wear_rate

class VehicleDataGenerator:
    def __init__(self):
        self.vehicle_profiles = {
            'VH001': {'type': 'Heavy Truck', 'base_speed': 60, 'load_factor': 1.2},
            'VH002': {'type': 'Delivery Van', 'base_speed': 45, 'load_factor': 0.8},
            'VH003': {'type': 'Long Haul', 'base_speed': 75, 'load_factor': 1.5},
            'VH004': {'type': 'City Bus', 'base_speed': 35, 'load_factor': 1.1},
            'VH005': {'type': 'Utility Truck', 'base_speed': 50, 'load_factor': 1.3}
        }
    
    def generate_vehicle_data(self, vehicle_id, num_records=1000):
        """Generate unique CSV data for a specific vehicle"""
        profile = self.vehicle_profiles.get(vehicle_id, {'base_speed': 60, 'load_factor': 1.0})
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=num_records),
            end=datetime.now(),
            freq='H'
        )
        
        data = []
        for i, timestamp in enumerate(timestamps[:num_records]):
            # Vehicle-specific patterns
            time_factor = np.sin(i / 50) * 0.3 + 0.7  # Cyclical pattern
            
            speed = max(0, np.random.normal(profile['base_speed'], 15) * time_factor)
            engine_load = max(0, min(100, np.random.normal(60, 20) * profile['load_factor']))
            strain = max(0, np.random.normal(150, 30) * profile['load_factor'])
            brake_pressure = max(0, np.random.normal(25, 10) * (1 + speed/100))
            brake_temp = max(0, np.random.normal(120, 40) + speed * 0.5)
            
            data.append({
                'Timestamp': timestamp,
                'VehicleID': vehicle_id,
                'Speed_kmh': speed,
                'EngineLoad_pct': engine_load,
                'Strain_micro': strain,
                'BrakePressure_bar': brake_pressure,
                'BrakeTemp_C': brake_temp,
                'RoadRoughness': max(0, np.random.normal(2.5, 1)),
                'SaltIndex': max(0, np.random.normal(0.3, 0.2)),
                'WindSpeed_kmh': max(0, np.random.normal(15, 8)),
                'FaultFlag': 1 if (brake_temp > 200 and speed > 80) else 0
            })
        
        return pd.DataFrame(data)

class AINavigationAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
        self.context = """
        You are BytEdge AI Assistant, a smart predictive maintenance agent for commercial vehicles.
        You help users navigate through maintenance dashboards, analyze vehicle data, and provide insights.
        
        Available Tabs:
        1. Fleet Overview - General fleet metrics and health
        2. Vehicle Analysis - Individual vehicle detailed analysis with live feed
        3. Predictive Analytics - RUL predictions and maintenance forecasts
        4. AI Assistant - This conversation interface
        
        You can help users:
        - Navigate to specific tabs
        - Analyze vehicle performance
        - Explain maintenance metrics
        - Generate insights from data
        - Set up alerts and monitoring
        """
    
    def process_query(self, query, current_tab="AI Assistant", vehicle_data=None):
        """Process user query and provide navigation or insights"""
        prompt = f"""
        {self.context}
        
        Current Active Tab: {current_tab}
        User Query: {query}
        
        Available Vehicle Data: {vehicle_data if vehicle_data else "No specific vehicle data provided"}
        
        Respond in a helpful, conversational manner. If the user wants to navigate somewhere, 
        suggest the appropriate tab and what they'll find there. If they're asking for analysis,
        provide insights based on typical vehicle maintenance patterns.
        
        Keep responses concise but informative. Use emojis where appropriate.
        """
        
        if not self.api_key:
            return "🔧 Please configure your Gemini API key in the sidebar to enable AI navigation."
        
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                }
            }
            
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"⚠️ API Error: {response.status_code}. Please check your API key."
                
        except Exception as e:
            return f"🔧 Connection error: {str(e)}"

def simulate_live_feed(data, points=50):
    """Simulate live data feed by gradually revealing data points"""
    if len(data) > points:
        return data.tail(points)
    return data

def main():
    st.markdown('<div class="main-header">🤖 BytEdge AI Predictive Maintenance Agent</div>', unsafe_allow_html=True)
    
    # Initialize engines
    pm_engine = PredictiveMaintenanceEngine()
    data_generator = VehicleDataGenerator()
    
    # Sidebar - AI Agent Configuration
    st.sidebar.markdown('<div class="ai-agent-container">', unsafe_allow_html=True)
    st.sidebar.title("🤖 AI Agent Configuration")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    ai_agent = AINavigationAgent(gemini_api_key)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Vehicle Selection
    st.sidebar.markdown('<div class="ai-agent-container">', unsafe_allow_html=True)
    st.sidebar.subheader("🚗 Vehicle Selection")
    vehicles = ['VH001', 'VH002', 'VH003', 'VH004', 'VH005']
    selected_vehicle = st.sidebar.selectbox("Choose Vehicle", vehicles)
    
    if st.sidebar.button("🔄 Generate Vehicle Data"):
        with st.spinner(f"Generating 1000 records for {selected_vehicle}..."):
            vehicle_data = data_generator.generate_vehicle_data(selected_vehicle, 1000)
            st.session_state.vehicle_data = vehicle_data
            st.session_state.current_vehicle = selected_vehicle
            st.success(f"✅ Generated 1000 records for {selected_vehicle}")
    
    # Download CSV
    if 'vehicle_data' in st.session_state:
        csv = st.session_state.vehicle_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.download_button(
            label="📥 Download Vehicle CSV",
            data=csv,
            file_name=f"{selected_vehicle}_data.csv",
            mime="text/csv"
        )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # AI Chat Interface in Sidebar
    st.sidebar.markdown('<div class="ai-agent-container">', unsafe_allow_html=True)
    st.sidebar.subheader("💬 AI Navigation Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.sidebar.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.sidebar.chat_input("Ask me about navigation or analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        vehicle_info = f"Current vehicle: {selected_vehicle}" if 'vehicle_data' in st.session_state else "No vehicle data loaded"
        response = ai_agent.process_query(prompt, "AI Assistant", vehicle_info)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.sidebar.chat_message("assistant"):
            st.markdown(response)
    
    if st.sidebar.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fleet Overview", 
        "🚗 Vehicle Analysis", 
        "🔮 Predictive Analytics", 
        "🤖 AI Assistant"
    ])
    
    with tab1:
        st.header("📊 Fleet Overview Dashboard")
        
        if 'vehicle_data' not in st.session_state:
            st.info("👆 Generate vehicle data from the sidebar to get started!")
            return
        
        df = st.session_state.vehicle_data
        current_vehicle = st.session_state.get('current_vehicle', 'VH001')
        
        # Calculate metrics
        df['FatigueIndex'] = df.apply(lambda x: pm_engine.calculate_fatigue_index(x['Strain_micro'], x['EngineLoad_pct']), axis=1)
        df['DragForce_N'] = df['Speed_kmh'].apply(pm_engine.calculate_drag_force)
        df['FuelRate_lph'] = df.apply(lambda x: pm_engine.calculate_fuel_rate(x['Speed_kmh'], x['EngineLoad_pct'], 10000, x['WindSpeed_kmh']), axis=1)
        df['BrakeWear_mm'] = df.apply(lambda x: pm_engine.calculate_brake_wear(x['BrakePressure_bar'], x['Speed_kmh'], x['BrakeTemp_C']), axis=1)
        
        cumulative_wear = df.groupby('VehicleID')['BrakeWear_mm'].cumsum()
        df['CumulativeWear_mm'] = cumulative_wear
        df['RUL_hours'] = df.apply(lambda x: pm_engine.calculate_rul(x['CumulativeWear_mm'], x['BrakeWear_mm']), axis=1)
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_speed = df['Speed_kmh'].mean()
            st.markdown(f'<div class="metric-card"><h4>🚀 Avg Speed</h4><p>{avg_speed:.1f} km/h</p></div>', unsafe_allow_html=True)
        
        with col2:
            fault_rate = df['FaultFlag'].mean() * 100
            risk_class = "risk-high" if fault_rate > 5 else "risk-medium" if fault_rate > 2 else "risk-low"
            st.markdown(f'<div class="metric-card"><h4>⚠️ Fault Rate</h4><p class="{risk_class}">{fault_rate:.1f}%</p></div>', unsafe_allow_html=True)
        
        with col3:
            avg_rul = df['RUL_hours'].mean()
            st.markdown(f'<div class="metric-card"><h4>⏱️ Avg RUL</h4><p>{avg_rul:.0f} hours</p></div>', unsafe_allow_html=True)
        
        with col4:
            efficiency = df['Speed_kmh'].sum() / df['FuelRate_lph'].sum()
            st.markdown(f'<div class="metric-card"><h4>⛽ Efficiency</h4><p>{efficiency:.1f} km/l</p></div>', unsafe_allow_html=True)
        
        # Fleet Health Overview
        st.subheader("🏥 Fleet Health Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(
                df, x='Speed_kmh', y='BrakeTemp_C', color='RUL_hours',
                title='Speed vs Brake Temperature',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.density_heatmap(
                df, x='EngineLoad_pct', y='FuelRate_lph',
                title='Engine Load vs Fuel Consumption',
                nbinsx=20, nbinsy=20
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header(f"🚗 Vehicle Analysis: {current_vehicle}")
        
        if 'vehicle_data' not in st.session_state:
            st.info("👆 Generate vehicle data first from the sidebar!")
            return
        
        df = st.session_state.vehicle_data
        
        # Live Feed Simulation
        st.subheader("📡 Live Data Feed Simulation")
        live_points = st.slider("Data Points to Show", 50, 200, 100)
        
        # Simulate live updates
        live_data = simulate_live_feed(df, live_points)
        
        # Create placeholder for live updates
        chart_placeholder = st.empty()
        
        # Live feed control
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎬 Start Live Feed"):
                st.session_state.live_feed = True
            if st.button("⏹️ Stop Live Feed"):
                st.session_state.live_feed = False
        
        with col2:
            if st.session_state.get('live_feed', False):
                st.markdown('<span class="live-feed-indicator"></span><strong>LIVE FEED ACTIVE</strong>', unsafe_allow_html=True)
        
        # Vehicle-specific metrics
        st.subheader(f"📈 {current_vehicle} Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.line(
                live_data, x='Timestamp', y='Speed_kmh',
                title=f'{current_vehicle} Speed Over Time',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            fig5 = px.line(
                live_data, x='Timestamp', y='BrakeTemp_C',
                title=f'{current_vehicle} Brake Temperature',
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            fig4 = px.line(
                live_data, x='Timestamp', y='EngineLoad_pct',
                title=f'{current_vehicle} Engine Load',
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            fig6 = px.line(
                live_data, x='Timestamp', y='Strain_micro',
                title=f'{current_vehicle} Chassis Strain',
                color_discrete_sequence=['#6bff6b']
            )
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.header("🔮 Predictive Analytics")
        
        if 'vehicle_data' not in st.session_state:
            st.info("👆 Generate vehicle data first from the sidebar!")
            return
        
        df = st.session_state.vehicle_data
        
        # RUL Predictions
        st.subheader("⏱️ Remaining Useful Life Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rul_dist = px.histogram(
                df, x='RUL_hours', nbins=30,
                title='RUL Distribution',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(rul_dist, use_container_width=True)
        
        with col2:
            wear_trend = px.line(
                df, x='Timestamp', y='CumulativeWear_mm',
                title='Brake Wear Progression',
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(wear_trend, use_container_width=True)
        
        # Risk Analysis
        st.subheader("⚠️ Risk Assessment")
        
        high_risk = (df['BrakeTemp_C'] > 180).sum()
        medium_risk = ((df['BrakeTemp_C'] > 150) & (df['BrakeTemp_C'] <= 180)).sum()
        low_risk = (df['BrakeTemp_C'] <= 150).sum()
        
        risk_data = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [high_risk, medium_risk, low_risk],
            'Color': ['#ff6b6b', '#ffd93d', '#6bff6b']
        })
        
        fig_risk = px.bar(
            risk_data, x='Risk Level', y='Count', color='Risk Level',
            title='Brake Temperature Risk Distribution',
            color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bff6b'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab4:
        st.header("🤖 AI Assistant Hub")
        
        st.markdown("""
        <div class="ai-agent-container">
        <h3>🎯 How I Can Help You:</h3>
        <p>• <strong>Navigate the app:</strong> "Take me to vehicle analysis"</p>
        <p>• <strong>Explain metrics:</strong> "What does RUL mean?"</p>
        <p>• <strong>Analyze data:</strong> "Show me vehicles with high risk"</p>
        <p>• <strong>Generate insights:</strong> "What maintenance is needed?"</p>
        <p>• <strong>Set up monitoring:</strong> "Create alerts for brake temperature"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Action Buttons
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Show Fleet Health"):
                st.session_state.messages.append({"role": "user", "content": "Show me fleet health overview"})
                st.rerun()
        
        with col2:
            if st.button("🔍 Analyze Current Vehicle"):
                st.session_state.messages.append({"role": "user", "content": f"Analyze {current_vehicle} performance"})
                st.rerun()
        
        with col3:
            if st.button("⚠️ Risk Assessment"):
                st.session_state.messages.append({"role": "user", "content": "Show me risk assessment"})
                st.rerun()
        
        # Current System Status
        st.subheader("📊 System Status")
        
        if 'vehicle_data' in st.session_state:
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                st.metric("Active Vehicle", current_vehicle)
            
            with status_col2:
                total_records = len(st.session_state.vehicle_data)
                st.metric("Data Records", f"{total_records:,}")
            
            with status_col3:
                fault_count = st.session_state.vehicle_data['FaultFlag'].sum()
                st.metric("Fault Events", fault_count)

if __name__ == "__main__":
    main()