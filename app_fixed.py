import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from database import get_db, create_user, verify_user, save_employee_data, get_employee_data
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
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction"])
    
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
        
        # Add a welcome image or other content instead of metrics
        st.image("https://img.freepik.com/free-vector/hr-management-abstract-concept_335657-3005.jpg", use_container_width=True)
        
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
        
        # Input form with tabs for organization
        tab1, tab2 = st.tabs(["Personal Information", "Professional Information"])
        
        with tab1:
            st.subheader("Personal Information")
            
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
        
        with tab2:
            st.subheader("Professional Information")
            
            # Monthly Income field
            monthly_income = st.number_input("Monthly Income", 
                                            min_value=100, max_value=5000000000 , value=5000, 
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
            st.markdown("##### Work Life Balance")
            work_life_balance = st.select_slider(
                "Select work-life balance rating",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Very Poor",
                    2: "2: Fair",
                    3: "3: Good",
                    4: "4: Excellent"
                }[x],
                value=3
            )
            st.info("Work Life Balance: Balance between professional and personal life affects overall satisfaction.")
            
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
            st.caption("Training Times Last Year: Number of training sessions the employee attended in the previous year.")
            
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
            st.caption("Environment Satisfaction: Employee's satisfaction with their work environment.")
        
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
            
            # Age factor (younger employees tend to leave more)
            if age < 30:
                risk_score += 0.2
            elif age > 50:
                risk_score += 0.1
                
            # Income factor (lower income increases attrition risk)
            if monthly_income < 3000:
                risk_score += 0.25
            elif monthly_income < 5000:
                risk_score += 0.15
                
            # Tenure factor (newer employees leave more often)
            if years_at_company < 2:
                risk_score += 0.2
            
            # Satisfaction factors
            if work_life_balance < 3:
                risk_score += 0.15
            if job_satisfaction < 3:
                risk_score += 0.2
            if environment_satisfaction < 3:
                risk_score += 0.1
                
            # Performance factor (lower performers might be more likely to leave)
            if performance_rating < 3:
                risk_score += 0.1
                
            # Distance factor
            if distance_from_home > 20:
                risk_score += 0.1
                
            # Normalize score between 0 and 1
            prediction = min(max(risk_score, 0.1), 0.9)  # Cap between 0.1 and 0.9
            
            # Update attrition in the database based on prediction
            employee_data['Attrition'] = prediction > 0.5
            
            # Display result
            st.subheader("Prediction Result")
            if prediction > 0.5:
                st.error(f"High risk of attrition ({prediction:.1%} probability)")
            else:
                st.success(f"Low risk of attrition ({prediction:.1%} probability)")
            
            # Save prediction to database
            save_employee_data(db, employee_data, st.session_state.user_id)
            
            # Display key factors affecting the prediction
            st.subheader("Key Factors Affecting Attrition Risk")
            
            # Create a dataframe of factors and their contribution to the risk score
            factors = [
                ("Age", 0.2 if age < 30 else (0.1 if age > 50 else 0)),
                ("Monthly Income", 0.25 if monthly_income < 3000 else (0.15 if monthly_income < 5000 else 0)),
                ("Years at Company", 0.2 if years_at_company < 2 else 0),
                ("Work Life Balance", 0.15 if work_life_balance < 3 else 0),
                ("Job Satisfaction", 0.2 if job_satisfaction < 3 else 0),
                ("Environment Satisfaction", 0.1 if environment_satisfaction < 3 else 0),
                ("Performance Rating", 0.1 if performance_rating < 3 else 0),
                ("Distance from Home", 0.1 if distance_from_home > 20 else 0)
            ]
            
            # Sort factors by their contribution
            factors.sort(key=lambda x: x[1], reverse=True)
            
            # Create dataframe for visualization
            factor_df = pd.DataFrame(factors, columns=['Factor', 'Contribution'])
            
            # Only show factors that contributed to the risk score
            factor_df = factor_df[factor_df['Contribution'] > 0]
            
            if len(factor_df) > 0:
                fig_factors = px.bar(factor_df, 
                                    x='Contribution', 
                                    y='Factor',
                                    orientation='h',
                                    title='Factors Contributing to Attrition Risk',
                                    labels={'Contribution': 'Impact on Risk Score', 'Factor': 'Employee Factor'})
                st.plotly_chart(fig_factors)
            else:
                st.info("No significant risk factors identified for this employee.")
