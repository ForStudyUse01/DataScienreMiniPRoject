import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from database import get_db, create_user, verify_user, save_employee_data, get_employee_data, save_prediction_history, get_prediction_history
import sqlalchemy.orm

# Set page config
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="👥",
    layout="wide"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Sidebar navigation
st.sidebar.title("Navigation")

# Login/Register section
if not st.session_state.logged_in:
    st.title("👥 Employee Attrition Analysis")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            db = get_db()
            user = verify_user(db, username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user.id
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                db = get_db()
                try:
                    user = create_user(db, new_username, new_password, email)
                    st.success("Registration successful! Please login.")
                except sqlalchemy.exc.IntegrityError:
                    st.error("Username or email already exists")

else:
    # Main application
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Prediction History"])
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.rerun()
    
    # Load data from database
    @st.cache_data
    def load_data():
        db = get_db()
        employees = get_employee_data(db, st.session_state.user_id)
        if not employees:
            # If no data exists, create sample data with all the new fields
            data = {
                'EmployeeName': [f'Employee {i}' for i in range(1, 101)],
                'Age': np.random.randint(20, 60, 100),
                'Gender': np.random.choice(['Male', 'Female'], 100),
                'RelationshipStatus': np.random.choice(['Single', 'Married', 'Divorced'], 100),
                'DistanceFromHome': np.random.randint(1, 30, 100),
                'MonthlyIncome': np.random.randint(2000, 15000, 100),
                'JobRole': np.random.choice(['Research Scientist', 'Sales Executive', 'Laboratory Technician', 
                                           'Manufacturing Director', 'Research Director', 'Sales Representative'], 100),
                'Department': np.random.choice(['Research & Development', 'Sales', 'Human Resources'], 100),
                'YearsAtCompany': np.random.randint(0, 20, 100),
                'WorkLifeBalance': np.random.randint(1, 5, 100),
                'JobLevel': np.random.choice(['Entry Level', 'Junior', 'Mid-Level', 'Senior', 'Executive'], 100),
                'TrainingTimesLastYear': np.random.randint(0, 6, 100),
                'JobSatisfaction': np.random.randint(1, 5, 100),
                'PerformanceRating': np.random.randint(1, 5, 100),
                'EnvironmentSatisfaction': np.random.randint(1, 5, 100),
                'Attrition': np.random.choice([False, True], 100, p=[0.8, 0.2])
            }
            df = pd.DataFrame(data)
            # Save sample data to database
            for _, row in df.iterrows():
                save_employee_data(db, row.to_dict(), st.session_state.user_id)
            return df
        else:
            # Convert database records to DataFrame with all the new fields
            data = {
                'EmployeeName': [e.employee_name for e in employees],
                'Age': [e.age for e in employees],
                'Gender': [e.gender for e in employees],
                'RelationshipStatus': [e.relationship_status for e in employees],
                'DistanceFromHome': [e.distance_from_home for e in employees],
                'MonthlyIncome': [e.monthly_income for e in employees],
                'JobRole': [e.job_role for e in employees],
                'Department': [e.department for e in employees],
                'YearsAtCompany': [e.years_at_company for e in employees],
                'WorkLifeBalance': [e.work_life_balance for e in employees],
                'JobLevel': [e.job_level for e in employees],
                'TrainingTimesLastYear': [e.training_times_last_year for e in employees],
                'JobSatisfaction': [e.job_satisfaction for e in employees],
                'PerformanceRating': [e.performance_rating for e in employees],
                'EnvironmentSatisfaction': [e.environment_satisfaction for e in employees],
                'Attrition': [e.attrition for e in employees]
            }
            return pd.DataFrame(data)
    
    df = load_data()
    
    if page == "Home":
        st.title("👥 Employee Attrition Analysis")
        st.write(f"""
        Welcome to the Employee Attrition Analysis Dashboard! This application helps you:
        
        - Analyze employee data and identify patterns
        - Predict employee attrition risk
        - Make data-driven decisions to improve retention
        
        Use the sidebar to navigate between different sections.
        """)
        
        # Add a spacer for better layout
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add some more detailed information about the application
        st.subheader("About This Application")
        st.write("""
        This HR Analytics application helps you analyze employee data and predict attrition risk. 
        
        **Key Features:**
        - **Data Analysis**: Visualize employee data to identify patterns and trends
        - **Attrition Prediction**: Predict which employees are at risk of leaving
        - **Employee Management**: Track and manage employee information
        
        Use the navigation menu on the left to explore different sections of the application.
        """)
    
    elif page == "Data Analysis":
        st.title("📊 Data Analysis")
        
        # Department-wise attrition
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].mean().reset_index()
        fig_dept = px.bar(dept_attrition, x='Department', y='Attrition',
                         title='Attrition Rate by Department')
        st.plotly_chart(fig_dept)
        
        # Age distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(df, x='Age', color='Attrition',
                              title='Age Distribution by Attrition Status')
        st.plotly_chart(fig_age)
        
        # Salary vs Years at Company
        st.subheader("Salary vs Years at Company")
        fig_salary = px.scatter(df, x='YearsAtCompany', y='MonthlyIncome',
                               color='Attrition', title='Monthly Income vs Years at Company')
        st.plotly_chart(fig_salary)
        
        # Job Satisfaction vs Work Life Balance
        st.subheader("Job Satisfaction vs Work Life Balance")
        fig_satisfaction = px.scatter(df, x='WorkLifeBalance', y='JobSatisfaction',
                                    color='Attrition', title='Job Satisfaction vs Work Life Balance',
                                    labels={'WorkLifeBalance': 'Work Life Balance (1-4)', 
                                            'JobSatisfaction': 'Job Satisfaction (1-4)'})
        st.plotly_chart(fig_satisfaction)
    
    elif page == "Prediction":
        st.title("🔮 Attrition Prediction")
        
        # Combined form with all fields
        st.subheader("Employee Information")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Employee Name field
            employee_name = st.text_input("Employee Name")
            
            # Age field
            age = st.number_input("Age", min_value=18, max_value=100, value=30, 
                                help="Employee's age in years")
            
            # Gender field
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            # Relationship Status field
            relationship_status = st.selectbox("Relationship Status", 
                                            ["Single", "Married", "Divorced"], 
                                            format_func=lambda x: {
                                                "Single": "1: Single",
                                                "Married": "2: Married", 
                                                "Divorced": "3: Divorced"
                                            }[x])
            
            # Distance from Home field
            distance_from_home = st.number_input("Distance From Home (km)", 
                                                min_value=0, value=10, 
                                                help="Distance between home and workplace in kilometers")
            
            # Monthly Income field
            monthly_income = st.number_input("Monthly Income", 
                                   min_value=100, max_value=500000000, value=5000, 
                                   help="Monthly salary in USD")
            
            # Job Role field
            job_role = st.selectbox("Job Role", [
                "Research Scientist", 
                "Sales Executive", 
                "Laboratory Technician", 
                "Manufacturing Director", 
                "Research Director", 
                "Sales Representative"
            ])
            
        with col2:
            # Department field
            department = st.selectbox("Department", [
                "Research & Development", 
                "Sales", 
                "Human Resources"
            ])
            
            # Years at Company field
            years_at_company = st.number_input("Years at Company", 
                                            min_value=0, max_value=40, value=3, 
                                            help="Number of years at the company")
            
            # Work Life Balance field with explanation
            work_life_balance = st.select_slider(
                "Work Life Balance",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Very Poor",
                    2: "2: Fair",
                    3: "3: Good",
                    4: "4: Excellent"
                }[x],
                value=3
            )
            st.caption("Balance between professional and personal life affects overall satisfaction.")
            
            # Job Level field with descriptive names
            job_level = st.selectbox("Job Level", [
                "Entry Level", 
                "Junior", 
                "Mid-Level", 
                "Senior", 
                "Executive"
            ])
            
            # Training Times Last Year field with explanation
            training_times_last_year = st.number_input("Training Times Last Year", 
                                                    min_value=0, max_value=10, value=2, 
                                                    help="Number of training sessions attended in the previous year")
            
            # Job Satisfaction field
            job_satisfaction = st.select_slider(
                "Job Satisfaction",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Extremely Unsatisfied",
                    2: "2: Somewhat Unsatisfied",
                    3: "3: Somewhat Satisfied",
                    4: "4: Extremely Satisfied"
                }[x],
                value=3
            )
            
            # Performance Rating field
            performance_rating = st.select_slider(
                "Performance Rating",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Needs Improvement",
                    2: "2: Meets Expectations",
                    3: "3: Exceeds Expectations",
                    4: "4: Outstanding"
                }[x],
                value=3
            )
            
            # Environment Satisfaction field with explanation
            environment_satisfaction = st.select_slider(
                "Environment Satisfaction",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Uncomfortable",
                    2: "2: Needs Improvement",
                    3: "3: Mostly Comfortable",
                    4: "4: Excellent Environment"
                }[x],
                value=3
            )
            st.caption("Employee's satisfaction with their work environment.")
        
        # Prediction button
        if st.button("Predict Attrition Risk"):
            # Convert relationship_status and job_level to numeric values for model
            relationship_map = {"Single": 1, "Married": 2, "Divorced": 3}
            job_level_map = {"Entry Level": 1, "Junior": 2, "Mid-Level": 3, "Senior": 4, "Executive": 5}
            
            # Prepare input data with all the new fields
            input_data = pd.DataFrame({
                'EmployeeName': [employee_name],
                'Age': [age],
                'Gender': [gender],
                'RelationshipStatus': [relationship_map.get(relationship_status, 1)],
                'DistanceFromHome': [distance_from_home],
                'MonthlyIncome': [monthly_income],
                'JobRole': [job_role],
                'Department': [department],
                'YearsAtCompany': [years_at_company],
                'WorkLifeBalance': [work_life_balance],
                'JobLevel': [job_level_map.get(job_level, 1)],
                'TrainingTimesLastYear': [training_times_last_year],
                'JobSatisfaction': [job_satisfaction],
                'PerformanceRating': [performance_rating],
                'EnvironmentSatisfaction': [environment_satisfaction]
            })
            
            # Save the employee data to the database
            db = get_db()
            employee_data = {
                'EmployeeName': employee_name,
                'Age': age,
                'Gender': gender,
                'RelationshipStatus': relationship_status,
                'DistanceFromHome': distance_from_home,
                'MonthlyIncome': monthly_income,
                'JobRole': job_role,
                'Department': department,
                'YearsAtCompany': years_at_company,
                'WorkLifeBalance': work_life_balance,
                'JobLevel': job_level,
                'TrainingTimesLastYear': training_times_last_year,
                'JobSatisfaction': job_satisfaction,
                'PerformanceRating': performance_rating,
                'EnvironmentSatisfaction': environment_satisfaction,
                'Attrition': False  # Default value, will be updated based on prediction
            }
            
            # One-hot encode categorical variables
            categorical_cols = ['Gender', 'Department', 'JobRole']
            input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
            
            # Calculate risk score based on known attrition factors
            risk_score = 0
            factor_weights = {}
            
            # Age factor (younger employees tend to leave more)
            if age < 30:
                risk_score += 0.18
                factor_weights["Age"] = 0.18
            elif age > 50:
                risk_score += 0.08
                factor_weights["Age"] = 0.08
            else:
                factor_weights["Age"] = 0.05
                risk_score += 0.05
                
            # Income factor (lower income increases attrition risk)
            if monthly_income < 3000:
                risk_score += 0.22
                factor_weights["Monthly Income"] = 0.22
            elif monthly_income < 5000:
                risk_score += 0.15
                factor_weights["Monthly Income"] = 0.15
            else:
                factor_weights["Monthly Income"] = 0.05
                risk_score += 0.05
                
            # Tenure factor (newer employees leave more often)
            if years_at_company < 2:
                risk_score += 0.17
                factor_weights["Years at Company"] = 0.17
            elif years_at_company < 5:
                risk_score += 0.10
                factor_weights["Years at Company"] = 0.10
            else:
                factor_weights["Years at Company"] = 0.05
                risk_score += 0.05
            
            # Work Life Balance factor
            if work_life_balance == 1:  # Very Poor
                risk_score += 0.20
                factor_weights["Work Life Balance"] = 0.20
            elif work_life_balance == 2:  # Fair
                risk_score += 0.12
                factor_weights["Work Life Balance"] = 0.12
            elif work_life_balance == 3:  # Good
                risk_score += 0.05
                factor_weights["Work Life Balance"] = 0.05
            else:  # Excellent
                factor_weights["Work Life Balance"] = 0.02
                risk_score += 0.02
                
            # Job Satisfaction factor
            if job_satisfaction == 1:  # Extremely Unsatisfied
                risk_score += 0.22
                factor_weights["Job Satisfaction"] = 0.22
            elif job_satisfaction == 2:  # Somewhat Unsatisfied
                risk_score += 0.15
                factor_weights["Job Satisfaction"] = 0.15
            elif job_satisfaction == 3:  # Somewhat Satisfied
                risk_score += 0.07
                factor_weights["Job Satisfaction"] = 0.07
            else:  # Extremely Satisfied
                factor_weights["Job Satisfaction"] = 0.02
                risk_score += 0.02
                
            # Environment Satisfaction factor
            if environment_satisfaction == 1:  # Uncomfortable
                risk_score += 0.15
                factor_weights["Environment Satisfaction"] = 0.15
            elif environment_satisfaction == 2:  # Needs Improvement
                risk_score += 0.10
                factor_weights["Environment Satisfaction"] = 0.10
            elif environment_satisfaction == 3:  # Mostly Comfortable
                risk_score += 0.05
                factor_weights["Environment Satisfaction"] = 0.05
            else:  # Excellent Environment
                factor_weights["Environment Satisfaction"] = 0.02
                risk_score += 0.02
                
            # Performance Rating factor
            if performance_rating == 1:  # Needs Improvement
                risk_score += 0.15
                factor_weights["Performance Rating"] = 0.15
            elif performance_rating == 2:  # Meets Expectations
                risk_score += 0.08
                factor_weights["Performance Rating"] = 0.08
            elif performance_rating == 3:  # Exceeds Expectations
                risk_score += 0.04
                factor_weights["Performance Rating"] = 0.04
            else:  # Outstanding
                factor_weights["Performance Rating"] = 0.02
                risk_score += 0.02
                
            # Distance factor
            if distance_from_home > 20:
                risk_score += 0.12
                factor_weights["Distance from Home"] = 0.12
            elif distance_from_home > 10:
                risk_score += 0.06
                factor_weights["Distance from Home"] = 0.06
            else:
                factor_weights["Distance from Home"] = 0.03
                risk_score += 0.03
                
            # Job Level factor
            job_level_map = {"Entry Level": 1, "Junior": 2, "Mid-Level": 3, "Senior": 4, "Executive": 5}
            job_level_numeric = job_level_map.get(job_level, 3)
            
            if job_level_numeric == 1:
                risk_score += 0.10
                factor_weights["Job Level"] = 0.10
            elif job_level_numeric == 2:
                risk_score += 0.07
                factor_weights["Job Level"] = 0.07
            elif job_level_numeric == 3:
                risk_score += 0.05
                factor_weights["Job Level"] = 0.05
            else:  # Senior or Executive
                factor_weights["Job Level"] = 0.03
                risk_score += 0.03
                
            # Normalize score between 0 and 1
            # Ensure prediction is at least 0.15 to avoid showing extremely low risk
            prediction = min(max(risk_score, 0.15), 0.9)  # Cap between 0.15 and 0.9
            
            # Update attrition in the database based on prediction
            employee_data['Attrition'] = prediction > 0.5
            
            # Display result with more detailed styling
            st.subheader("Prediction Result")
            
            # Create columns for better layout
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                # Display the probability as a large number
                if prediction > 0.5:
                    st.markdown(f"<h1 style='color: #ff4b4b; text-align: center;'>{prediction:.1%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #ff4b4b; text-align: center; font-weight: bold;'>High risk of attrition</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='color: #00c851; text-align: center;'>{prediction:.1%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #00c851; text-align: center; font-weight: bold;'>Low risk of attrition</p>", unsafe_allow_html=True)
            
            with result_col2:
                # Create a gauge chart to visualize the risk
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Attrition Risk", 'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                        'bar': {'color': "#ff4b4b" if prediction > 0.5 else "#00c851"},
                        'bgcolor': "rgba(50,50,50,0.8)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(0, 200, 81, 0.5)'},
                            {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.5)'},
                            {'range': [70, 100], 'color': 'rgba(255, 75, 75, 0.5)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Save prediction to database
            save_employee_data(db, employee_data, st.session_state.user_id)
            
            # Save to prediction history
            prediction_data = {
                'EmployeeName': employee_name,
                'Age': age,
                'Gender': gender,
                'RelationshipStatus': relationship_status,
                'DistanceFromHome': distance_from_home,
                'MonthlyIncome': monthly_income,
                'JobRole': job_role,
                'Department': department,
                'YearsAtCompany': years_at_company,
                'WorkLifeBalance': work_life_balance,
                'JobLevel': job_level,
                'TrainingTimesLastYear': training_times_last_year,
                'JobSatisfaction': job_satisfaction,
                'PerformanceRating': performance_rating,
                'EnvironmentSatisfaction': environment_satisfaction,
                'AttritionRisk': prediction
            }
            save_prediction_history(db, prediction_data, st.session_state.user_id)
            
            # Display key factors affecting the prediction
            st.subheader("Key Factors Affecting Attrition Risk")
            
            # Create a dataframe of factors and their contribution to the risk score
            factors = [(factor, weight) for factor, weight in factor_weights.items()]
            
            # Sort factors by their contribution
            factors.sort(key=lambda x: x[1], reverse=True)
            
            # Create dataframe for visualization
            factor_df = pd.DataFrame(factors, columns=['Factor', 'Contribution'])
            
            # Always show at least some factors, even if minimal
            # If no significant factors, add a minimal default factor
            if len(factor_df[factor_df['Contribution'] > 0]) == 0:
                factor_df = pd.DataFrame([
                    ("Age", 0.05),
                    ("Monthly Income", 0.03),
                    ("Years at Company", 0.02)
                ], columns=['Factor', 'Contribution'])
            else:
                factor_df = factor_df[factor_df['Contribution'] > 0]
            
            # Create the visualization with improved styling
            fig_factors = px.bar(factor_df, 
                                x='Contribution', 
                                y='Factor',
                                orientation='h',
                                title='Factors Contributing to Attrition Risk',
                                labels={'Contribution': 'Impact on Risk Score', 'Factor': 'Employee Factor'},
                                color='Contribution',
                                color_continuous_scale='reds')
            
            # Customize the layout
            fig_factors.update_layout(
                plot_bgcolor='rgba(0,0,0,0.05)',
                paper_bgcolor='rgba(0,0,0,0.05)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_showscale=False
            )
            
            # Update axes
            fig_factors.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            fig_factors.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig_factors, use_container_width=True)
    
    elif page == "Prediction History":
        st.title("📊 Prediction History")
        
        # Get prediction history from database
        db = get_db()
        history = get_prediction_history(db, st.session_state.user_id)
        
        if not history:
            st.info("No prediction history found. Make some predictions first!")
        else:
            st.write(f"Showing {len(history)} most recent predictions.")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Table View", "Chart View"])
            
            with tab1:
                # Create a dataframe for the history table
                history_data = {
                    'Date': [h.prediction_date.strftime("%Y-%m-%d %H:%M") for h in history],
                    'Employee': [h.employee_name for h in history],
                    'Department': [h.department for h in history],
                    'Job Role': [h.job_role for h in history],
                    'Age': [h.age for h in history],
                    'Years at Company': [h.years_at_company for h in history],
                    'Risk Score': [f"{h.attrition_risk:.1%}" for h in history],
                    'Risk Level': ["High" if h.attrition_risk > 0.5 else "Low" for h in history]
                }
                
                history_df = pd.DataFrame(history_data)
                
                # Add styling to the table
                def highlight_risk(val):
                    if val == "High":
                        return 'background-color: rgba(255, 75, 75, 0.2)'
                    else:
                        return 'background-color: rgba(0, 200, 81, 0.2)'
                
                # Display the styled table
                st.dataframe(
                    history_df.style.applymap(highlight_risk, subset=['Risk Level']),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Option to download the history as CSV
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download History as CSV",
                    data=csv,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )
            
            with tab2:
                # Create visualizations of the prediction history
                st.subheader("Risk Distribution")
                
                # Create risk distribution chart
                risk_counts = {
                    "High Risk": sum(1 for h in history if h.attrition_risk > 0.5),
                    "Low Risk": sum(1 for h in history if h.attrition_risk <= 0.5)
                }
                
                fig_risk = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    title="Distribution of Risk Levels",
                    color=list(risk_counts.keys()),
                    color_discrete_map={"High Risk": "#ff4b4b", "Low Risk": "#00c851"}
                )
                
                fig_risk.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    paper_bgcolor='rgba(0,0,0,0.05)',
                    font=dict(color='white'),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Create department-wise risk chart
                st.subheader("Department-wise Risk")
                
                dept_data = {}
                for h in history:
                    if h.department not in dept_data:
                        dept_data[h.department] = []
                    dept_data[h.department].append(h.attrition_risk)
                
                dept_avg_risk = {
                    dept: sum(risks)/len(risks) for dept, risks in dept_data.items() if risks
                }
                
                dept_df = pd.DataFrame({
                    'Department': list(dept_avg_risk.keys()),
                    'Average Risk': list(dept_avg_risk.values())
                })
                
                fig_dept = px.bar(
                    dept_df,
                    x='Department',
                    y='Average Risk',
                    title="Average Attrition Risk by Department",
                    color='Average Risk',
                    color_continuous_scale='reds',
                    text_auto='.1%'
                )
                
                fig_dept.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    paper_bgcolor='rgba(0,0,0,0.05)',
                    font=dict(color='white'),
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_showscale=False
                )
                
                fig_dept.update_traces(textposition='outside')
                fig_dept.update_yaxes(tickformat='.0%')
                
                st.plotly_chart(fig_dept, use_container_width=True)
                
                # Create timeline of predictions
                st.subheader("Prediction Timeline")
                
                timeline_data = {
                    'Date': [h.prediction_date for h in history],
                    'Risk Score': [h.attrition_risk for h in history],
                    'Employee': [h.employee_name for h in history]
                }
                
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df = timeline_df.sort_values('Date')
                
                fig_timeline = px.line(
                    timeline_df,
                    x='Date',
                    y='Risk Score',
                    markers=True,
                    hover_data=['Employee'],
                    title="Attrition Risk Timeline",
                    color_discrete_sequence=["#1976d2"]
                )
                
                fig_timeline.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    paper_bgcolor='rgba(0,0,0,0.05)',
                    font=dict(color='white'),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                fig_timeline.update_yaxes(tickformat='.0%')
                
                st.plotly_chart(fig_timeline, use_container_width=True)
