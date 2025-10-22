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

# Page configuration
st.set_page_config(
    page_title="BytEdge Predictive Maintenance",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        color:black;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class PredictiveMaintenanceEngine:
    def __init__(self):
        self.material_constants = {
            'fatigue_k': 1.2e-8,
            'wear_k': 3.5e-6,
            'drag_coefficient': 0.65,
            'frontal_area': 8.5,  # m²
            'air_density': 1.225,  # kg/m³
            'brake_pad_thickness_new': 12.0  # mm
        }
    
    def calculate_fatigue_index(self, strain, engine_load):
        """Calculate fatigue index using Miner's Rule approximation"""
        return self.material_constants['fatigue_k'] * (strain ** 2) * engine_load
    
    def calculate_drag_force(self, speed_kmh):
        """Calculate aerodynamic drag force"""
        speed_ms = speed_kmh / 3.6
        return 0.5 * self.material_constants['air_density'] * \
               self.material_constants['drag_coefficient'] * \
               self.material_constants['frontal_area'] * \
               (speed_ms ** 2)
    
    def calculate_fuel_rate(self, speed_kmh, engine_load, payload_kg, wind_speed):
        """Calculate fuel rate based on multiple factors"""
        base_rate = 8.0  # lph
        speed_effect = 0.02 * (speed_kmh ** 2) * self.material_constants['drag_coefficient']
        load_effect = 0.001 * engine_load * payload_kg / 1000
        wind_effect = 0.1 * abs(wind_speed)
        
        return base_rate + speed_effect + load_effect + wind_effect + np.random.normal(0, 0.5)
    
    def calculate_brake_wear(self, brake_pressure, speed_kmh, brake_temp):
        """Calculate brake wear rate"""
        braking_work = brake_pressure * (speed_kmh / 3.6) * 0.01
        temp_factor = 1 + 0.002 * max(0, brake_temp - 100)
        return self.material_constants['wear_k'] * braking_work * temp_factor
    
    def calculate_rul(self, current_wear, wear_rate):
        """Calculate remaining useful life"""
        remaining_material = self.material_constants['brake_pad_thickness_new'] - current_wear
        if wear_rate <= 0:
            return float('inf')
        return remaining_material / wear_rate

class GeneticAlgorithmOptimizer:
    def __init__(self):
        self.param_bounds = {
            'brake_pad_area': (50, 200),  # cm²
            'actuator_diameter': (20, 60),  # mm
            'disc_ventilation': (1, 10),  # ventilation factor
            'friction_coefficient': (0.3, 0.6)
        }
    
    def fitness_function(self, individual):
        """Multi-objective fitness function for brake system optimization"""
        brake_pad_area, actuator_diameter, disc_vent, friction_coefficient = individual
        
        # Normalize parameters
        brake_force = brake_pad_area * friction_coefficient * 0.1
        pedal_effort = 100 / (actuator_diameter * 0.5)
        cooling_efficiency = disc_vent * 0.8
        wear_resistance = friction_coefficient * 0.7 + brake_pad_area * 0.001
        
        # Combined fitness (maximize brake force and cooling, minimize pedal effort)
        fitness = (brake_force * 0.4 + cooling_efficiency * 0.3 + 
                  wear_resistance * 0.2 - pedal_effort * 0.1)
        
        return fitness,
    
    def run_optimization(self, population_size=50, generations=100):
        """Run genetic algorithm optimization"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_float, toolbox.attr_float, 
                         toolbox.attr_float, toolbox.attr_float), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self.fitness_function)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=population_size)
        
        for gen in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, k=len(population))
        
        best_individual = tools.selBest(population, k=1)[0]
        return best_individual

class GeminiAIIntegration:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    
    def get_insights(self, data_summary, question=None):
        """Get AI-generated insights using Gemini API"""
        if not self.api_key:
            return "Gemini API key not configured. Please add your API key to enable AI insights."
        
        if question:
            prompt = f"""
            Based on the following vehicle maintenance data summary, answer this question: {question}
            
            Data Summary:
            {data_summary}
            
            Provide a concise, technical answer focused on predictive maintenance insights.
            """
        else:
            prompt = f"""
            Analyze this commercial vehicle fleet maintenance data and provide key insights:
            
            {data_summary}
            
            Focus on:
            1. Critical maintenance issues
            2. Efficiency optimization opportunities
            3. Risk factors and recommendations
            4. Predictive maintenance suggestions
            
            Keep response concise and actionable.
            """
        
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
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
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"Error getting AI insights: {str(e)}"

def generate_sample_data(num_vehicles=10, days=30):
    """Generate realistic sample data for demonstration"""
    vehicles = [f"VH{str(i).zfill(3)}" for i in range(1, num_vehicles + 1)]
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='H'
    )
    
    data = []
    for vehicle in vehicles:
        for timestamp in timestamps:
            speed = np.random.normal(65, 20)
            engine_load = np.random.normal(60, 20)
            strain = np.random.normal(150, 30)
            brake_pressure = np.random.normal(25, 10)
            brake_temp = np.random.normal(120, 40)
            road_roughness = np.random.normal(2.5, 1)
            salt_index = np.random.normal(0.3, 0.2)
            wind_speed = np.random.normal(15, 8)
            
            data.append({
                'Timestamp': timestamp,
                'VehicleID': vehicle,
                'Speed_kmh': max(0, speed),
                'EngineLoad_pct': max(0, min(100, engine_load)),
                'Strain_micro': max(0, strain),
                'BrakePressure_bar': max(0, brake_pressure),
                'BrakeTemp_C': max(0, brake_temp),
                'RoadRoughness': max(0, road_roughness),
                'SaltIndex': max(0, salt_index),
                'WindSpeed_kmh': max(0, wind_speed),
                'FaultFlag': np.random.choice([0, 1], p=[0.95, 0.05])
            })
    
    return pd.DataFrame(data)

def main():
    st.markdown('<div class="main-header">🚛 BytEdge Predictive Maintenance Demo</div>', 
                unsafe_allow_html=True)
    
    # Initialize engines
    pm_engine = PredictiveMaintenanceEngine()
    ga_optimizer = GeneticAlgorithmOptimizer()
    
    # Sidebar
    st.sidebar.title("Configuration")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    ai_integration = GeminiAIIntegration(gemini_api_key)
    
    # Data generation
    st.sidebar.subheader("Data Settings")
    num_vehicles = st.sidebar.slider("Number of Vehicles", 5, 20, 10)
    simulation_days = st.sidebar.slider("Simulation Days", 7, 90, 30)
    
    if st.sidebar.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            df = generate_sample_data(num_vehicles, simulation_days)
            st.session_state.df = df
            st.success(f"Generated {len(df)} records for {num_vehicles} vehicles")
    
    if 'df' not in st.session_state:
        st.info("Click 'Generate Sample Data' in the sidebar to start the demo")
        return
    
    df = st.session_state.df
    
    # Calculate derived metrics
    df['FatigueIndex'] = df.apply(
        lambda x: pm_engine.calculate_fatigue_index(x['Strain_micro'], x['EngineLoad_pct']), 
        axis=1
    )
    df['DragForce_N'] = df['Speed_kmh'].apply(pm_engine.calculate_drag_force)
    df['FuelRate_lph'] = df.apply(
        lambda x: pm_engine.calculate_fuel_rate(x['Speed_kmh'], x['EngineLoad_pct'], 10000, x['WindSpeed_kmh']), 
        axis=1
    )
    df['BrakeWear_mm'] = df.apply(
        lambda x: pm_engine.calculate_brake_wear(x['BrakePressure_bar'], x['Speed_kmh'], x['BrakeTemp_C']), 
        axis=1
    )
    
    # Calculate cumulative wear and RUL
    cumulative_wear = df.groupby('VehicleID')['BrakeWear_mm'].cumsum()
    df['CumulativeWear_mm'] = cumulative_wear
    df['RUL_hours'] = df.apply(
        lambda x: pm_engine.calculate_rul(x['CumulativeWear_mm'], x['BrakeWear_mm']), 
        axis=1
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Chassis Durability", 
        "🌬️ Aero & Fuel", 
        "🛑 Brake System", 
        "📊 Fleet Intelligence"
    ])
    
    with tab1:
        st.header("Chassis Durability Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_fatigue = df['FatigueIndex'].mean()
            st.metric("Average Fatigue Index", f"{avg_fatigue:.2e}")
        
        with col2:
            high_fatigue_vehicles = df[df['FatigueIndex'] > df['FatigueIndex'].quantile(0.9)]['VehicleID'].nunique()
            st.metric("Vehicles with High Fatigue", high_fatigue_vehicles)
        
        with col3:
            strain_correlation = df[['Strain_micro', 'RoadRoughness']].corr().iloc[0,1]
            st.metric("Strain-Road Roughness Correlation", f"{strain_correlation:.3f}")
        
        # Plots
        fig1 = px.scatter(
            df, x='RoadRoughness', y='Strain_micro', color='VehicleID',
            title='Strain vs Road Roughness',
            labels={'RoadRoughness': 'Road Roughness Index', 'Strain_micro': 'Strain (micro)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.line(
                df[df['VehicleID'].isin(df['VehicleID'].unique()[:3])], 
                x='Timestamp', y='FatigueIndex', color='VehicleID',
                title='Fatigue Index Over Time (Sample Vehicles)'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fatigue_by_vehicle = df.groupby('VehicleID')['FatigueIndex'].mean().sort_values(ascending=False)
            fig3 = px.bar(
                x=fatigue_by_vehicle.index, y=fatigue_by_vehicle.values,
                title='Average Fatigue Index by Vehicle',
                labels={'x': 'Vehicle ID', 'y': 'Fatigue Index'}
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.header("Aerodynamic & Fuel Efficiency Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_fuel_rate = df['FuelRate_lph'].mean()
            st.metric("Average Fuel Rate", f"{avg_fuel_rate:.1f} lph")
        
        with col2:
            avg_drag_force = df['DragForce_N'].mean()
            st.metric("Average Drag Force", f"{avg_drag_force:.0f} N")
        
        with col3:
            fuel_efficiency = df['Speed_kmh'].sum() / df['FuelRate_lph'].sum()
            st.metric("Fuel Efficiency", f"{fuel_efficiency:.1f} km/l")
        
        # Fuel efficiency analysis
        fig4 = px.scatter(
            df, x='Speed_kmh', y='FuelRate_lph', color='DragForce_N',
            title='Fuel Rate vs Speed (colored by Drag Force)',
            labels={'Speed_kmh': 'Speed (km/h)', 'FuelRate_lph': 'Fuel Rate (l/h)'}
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            wind_effect = px.density_heatmap(
                df, x='WindSpeed_kmh', y='FuelRate_lph',
                title='Wind Speed vs Fuel Rate Density',
                nbinsx=20, nbinsy=20
            )
            st.plotly_chart(wind_effect, use_container_width=True)
        
        with col2:
            speed_bins = pd.cut(df['Speed_kmh'], bins=10)
            efficiency_by_speed = df.groupby(speed_bins)['FuelRate_lph'].mean()
            fig5 = px.line(
                x=[bin.mid for bin in efficiency_by_speed.index],
                y=efficiency_by_speed.values,
                title='Optimal Speed for Fuel Efficiency',
                labels={'x': 'Speed (km/h)', 'y': 'Average Fuel Rate (l/h)'}
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.header("Brake System Risk Analysis")
        
        # Calculate risk metrics
        high_temp_risk = (df['BrakeTemp_C'] > 200).mean() * 100
        high_pressure_risk = (df['BrakePressure_bar'] > 40).mean() * 100
        low_rul_risk = (df['RUL_hours'] < 100).mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("High Temperature Risk", f"{high_temp_risk:.1f}%")
        
        with col2:
            st.metric("High Pressure Risk", f"{high_pressure_risk:.1f}%")
        
        with col3:
            st.metric("Low RUL Risk", f"{low_rul_risk:.1f}%")
        
        with col4:
            avg_rul = df['RUL_hours'].mean()
            st.metric("Average RUL", f"{avg_rul:.0f} hours")
        
        # Brake system visualizations
        fig6 = px.scatter(
            df, x='BrakePressure_bar', y='BrakeTemp_C', color='RUL_hours',
            title='Brake Pressure vs Temperature (colored by RUL)',
            color_continuous_scale='RdYlGn_r',
            labels={'BrakePressure_bar': 'Brake Pressure (bar)', 
                   'BrakeTemp_C': 'Brake Temperature (°C)'}
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            rul_distribution = px.histogram(
                df, x='RUL_hours', nbins=50,
                title='RUL Distribution Across Fleet',
                labels={'RUL_hours': 'Remaining Useful Life (hours)'}
            )
            st.plotly_chart(rul_distribution, use_container_width=True)
        
        with col2:
            wear_trend = df.groupby('VehicleID')['CumulativeWear_mm'].last().sort_values()
            fig7 = px.bar(
                x=wear_trend.index, y=wear_trend.values,
                title='Total Brake Wear by Vehicle',
                labels={'x': 'Vehicle ID', 'y': 'Cumulative Wear (mm)'}
            )
            st.plotly_chart(fig7, use_container_width=True)
        
        # Genetic Algorithm Optimization
        st.subheader("Brake System Optimization")
        if st.button("Run Genetic Algorithm Optimization"):
            with st.spinner("Running optimization..."):
                best_solution = ga_optimizer.run_optimization(population_size=30, generations=50)
                
                st.success("Optimization completed!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Optimal Brake Pad Area", f"{best_solution[0]:.1f} cm²")
                with col2:
                    st.metric("Optimal Actuator Diameter", f"{best_solution[1]:.1f} mm")
                with col3:
                    st.metric("Optimal Disc Ventilation", f"{best_solution[2]:.1f}")
                with col4:
                    st.metric("Optimal Friction Coefficient", f"{best_solution[3]:.3f}")
    
    with tab4:
        st.header("Fleet Intelligence Overview")
        
        # KPI Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vehicles = df['VehicleID'].nunique()
            st.metric("Total Vehicles", total_vehicles)
        
        with col2:
            fault_rate = df['FaultFlag'].mean() * 100
            risk_class = "risk-high" if fault_rate > 5 else "risk-medium" if fault_rate > 2 else "risk-low"
            st.markdown(f'<div class="kpi-card">Fault Rate: <span class="{risk_class}">{fault_rate:.1f}%</span></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            avg_efficiency = df['Speed_kmh'].sum() / df['FuelRate_lph'].sum()
            st.metric("Fleet Avg Efficiency", f"{avg_efficiency:.1f} km/l")
        
        with col4:
            maintenance_urgency = (df['RUL_hours'] < 50).mean() * 100
            risk_class = "risk-high" if maintenance_urgency > 10 else "risk-medium" if maintenance_urgency > 5 else "risk-low"
            st.markdown(f'<div class="kpi-card">Maintenance Urgency: <span class="{risk_class}">{maintenance_urgency:.1f}%</span></div>', 
                       unsafe_allow_html=True)
        
        # Fleet overview visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_health = df.groupby('VehicleID').agg({
                'FatigueIndex': 'mean',
                'RUL_hours': 'mean',
                'FaultFlag': 'mean'
            }).reset_index()
            
            fig8 = px.scatter(
                vehicle_health, x='FatigueIndex', y='RUL_hours', 
                size='FaultFlag', color='FaultFlag',
                hover_data=['VehicleID'],
                title='Vehicle Health Matrix: Fatigue vs RUL',
                labels={'FatigueIndex': 'Fatigue Index', 'RUL_hours': 'Remaining Useful Life'}
            )
            st.plotly_chart(fig8, use_container_width=True)
        
        with col2:
            performance_by_vehicle = df.groupby('VehicleID').agg({
                'FuelRate_lph': 'mean',
                'Speed_kmh': 'mean',
                'BrakeWear_mm': 'mean'
            }).reset_index()
            
            fig9 = px.parallel_coordinates(
                performance_by_vehicle,
                color='FuelRate_lph',
                dimensions=['FuelRate_lph', 'Speed_kmh', 'BrakeWear_mm'],
                title='Vehicle Performance Parallel Coordinates'
            )
            st.plotly_chart(fig9, use_container_width=True)
        
        # AI Insights Section
        st.subheader("AI-Powered Insights")
        
        # Data summary for AI
        data_summary = f"""
        Fleet Size: {df['VehicleID'].nunique()} vehicles
        Data Period: {df['Timestamp'].min()} to {df['Timestamp'].max()}
        Average Fatigue Index: {df['FatigueIndex'].mean():.2e}
        Average Fuel Rate: {df['FuelRate_lph'].mean():.1f} lph
        Average RUL: {df['RUL_hours'].mean():.0f} hours
        Fault Rate: {df['FaultFlag'].mean() * 100:.1f}%
        High Risk Vehicles (RUL < 100h): {(df['RUL_hours'] < 100).mean() * 100:.1f}%
        """
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Fleet Insights"):
                with st.spinner("Getting AI insights..."):
                    insights = ai_integration.get_insights(data_summary)
                    st.text_area("AI Insights", insights, height=00)
        
        with col2:
            st.subheader("Ask AI")
            question = st.text_input("Ask a question about your fleet data:")
            if question and st.button("Get Answer"):
                with st.spinner("Thinking..."):
                    answer = ai_integration.get_insights(data_summary, question)
                    st.text_area("AI Response", answer, height=150)

if __name__ == "__main__":
    main()
