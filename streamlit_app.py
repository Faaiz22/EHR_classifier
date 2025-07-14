import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import train_model, classify_result_clinical, get_prediction, calculate_clinical_reference_range
import numpy as np

# Set page config
st.set_page_config(
    page_title="EHR Lab Test Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the EHR dataset"""
    try:
        data = pd.read_csv('data-ori.csv')
        return data
    except FileNotFoundError:
        st.error("❌ Data file 'data-ori.csv' not found. Please ensure the file is in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def create_clinical_visualization(data, test_name, age, sex, test_value, reference_range):
    """Create scatter plot with clinical reference ranges and user input highlighted"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter data by sex
    male_data = data[data['SEX'] == 'M']
    female_data = data[data['SEX'] == 'F']
    
    # Create scatter plots
    ax.scatter(male_data['AGE'], male_data[test_name], alpha=0.6, color='blue', label='Male', s=50)
    ax.scatter(female_data['AGE'], female_data[test_name], alpha=0.6, color='red', label='Female', s=50)
    
    # Calculate and plot clinical reference ranges by age groups
    age_groups = [(0, 18), (18, 30), (30, 50), (50, 70), (70, 120)]
    
    for sex_val, color in [('M', 'blue'), ('F', 'red')]:
        for age_min, age_max in age_groups:
            # Get data for this age group and sex
            age_sex_data = data[(data['SEX'] == sex_val) & 
                               (data['AGE'] >= age_min) & 
                               (data['AGE'] < age_max)][test_name]
            
            if len(age_sex_data) >= 10:  # Minimum sample size
                # Calculate 2.5th and 97.5th percentiles (clinical normal range)
                lower = age_sex_data.quantile(0.025)
                upper = age_sex_data.quantile(0.975)
                mean_age = (age_min + age_max) / 2
                
                # Plot reference range as error bars
                ax.errorbar(mean_age, age_sex_data.median(), 
                          yerr=[[age_sex_data.median() - lower], [upper - age_sex_data.median()]], 
                          fmt='s', color=color, alpha=0.7, capsize=5, capthick=2,
                          label=f'{sex_val} Reference Range' if age_min == 0 else "")
    
    # Highlight user input
    user_color = 'blue' if sex == 'M' else 'red'
    ax.scatter([age], [test_value], color=user_color, s=300, marker='*', 
              edgecolor='black', linewidth=3, label=f'Your Input ({sex})', zorder=5)
    
    # Add reference range for user's age/sex as horizontal lines
    if reference_range:
        ax.axhline(y=reference_range['lower'], color='green', linestyle='--', alpha=0.8, 
                  label='Your Normal Range')
        ax.axhline(y=reference_range['upper'], color='green', linestyle='--', alpha=0.8)
        
        # Fill the normal range area
        ax.fill_between([data['AGE'].min(), data['AGE'].max()], 
                       reference_range['lower'], reference_range['upper'], 
                       color='green', alpha=0.1, label='Your Normal Range')
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(f'{test_name} Value', fontsize=12)
    ax.set_title(f'{test_name} vs Age - Clinical Reference Ranges', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set style
    sns.despine()
    plt.tight_layout()
    
    return fig

def main():
    """Main application function"""
    st.title("🔬 EHR Lab Test Analysis")
    st.markdown("### Analyze lab test results using regression models to detect abnormal values")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Define lab tests
    lab_tests = [
        'HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 
        'THROMBOCYTE', 'MCH', 'MCHC', 'MCV'
    ]
    
    # Data preprocessing
    # Convert SEX to numeric (M=1, F=0)
    data['SEX_M'] = (data['SEX'] == 'M').astype(int)
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🔧 Input Parameters")
        
        # Test selection
        selected_test = st.selectbox(
            "Select Lab Test:",
            lab_tests,
            help="Choose which lab test to analyze"
        )
        
        # Age input
        age = st.number_input(
            "Age (years):",
            min_value=0,
            max_value=120,
            value=30,
            help="Enter patient age"
        )
        
        # Sex selection
        sex = st.selectbox(
            "Sex:",
            ['M', 'F'],
            help="Select patient sex"
        )
        
        # Test value input
        test_value = st.number_input(
            f"{selected_test} Value:",
            value=float(data[selected_test].mean()),
            help=f"Enter the {selected_test} test result"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Result", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Analysis for {selected_test}")
        
        # Show dataset info
        with st.expander("📋 Dataset Information"):
            st.write(f"**Dataset Shape:** {data.shape}")
            st.write(f"**Selected Test Stats:**")
            st.write(data[selected_test].describe())
        
        if analyze_button:
            with st.spinner("Training model and analyzing..."):
                try:
                    # Train model
                    model = train_model(data, selected_test)
                    
                    # Get prediction
                    sex_numeric = 1 if sex == 'M' else 0
                    prediction = get_prediction(model, age, sex_numeric)
                    
                    # Classify result using clinical reference ranges
                    classification, percentile, reference_range = classify_result_clinical(
                        data, selected_test, age, sex, test_value
                    )
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Results metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Model Prediction",
                            f"{prediction:.2f}",
                            help="Expected value based on age and sex"
                        )
                    
                    with col_b:
                        st.metric(
                            "Percentile",
                            f"{percentile:.1f}%",
                            help="Percentile within age/sex group"
                        )
                    
                    with col_c:
                        st.metric(
                            "Reference Range",
                            f"{reference_range['lower']:.1f} - {reference_range['upper']:.1f}",
                            help="Clinical normal range for age/sex"
                        )
                    
                    # Classification result
                    if classification == "Normal":
                        st.success(f"🟢 **Result: {classification}**")
                        st.info(f"The test result is within the clinical normal range for {age}-year-old {sex.lower()}ale patients.")
                    else:
                        st.error(f"🔴 **Result: {classification}**")
                        st.warning(f"The test result is outside the clinical normal range for {age}-year-old {sex.lower()}ale patients.")
                    
                    # Show reference range details
                    st.write("**Clinical Reference Range:**")
                    st.write(f"- Lower limit: {reference_range['lower']:.2f}")
                    st.write(f"- Upper limit: {reference_range['upper']:.2f}")
                    st.write(f"- Your value: {test_value:.2f}")
                    st.write(f"- Percentile: {percentile:.1f}%")
                    
                    # Create and display visualization
                    fig = create_clinical_visualization(data, selected_test, age, sex, test_value, reference_range)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        st.subheader("📈 Model Information")
        
        if analyze_button:
            try:
                # Calculate clinical reference range for user's age/sex
                reference_range = calculate_clinical_reference_range(data, selected_test, age, sex)
                
                # Display reference range information
                st.write("**Clinical Reference Range:**")
                st.write(f"- Age Group: {reference_range['age_group']}")
                st.write(f"- Sex: {sex}")
                st.write(f"- Normal Range: {reference_range['lower']:.2f} - {reference_range['upper']:.2f}")
                st.write(f"- Sample Size: {reference_range['sample_size']}")
                
                # Show percentile information
                percentile = reference_range['percentile']
                st.write(f"- Your Value Percentile: {percentile:.1f}%")
                
            except Exception as e:
                st.error(f"Error calculating reference range: {str(e)}")
        else:
            st.info("👈 Select parameters and click 'Analyze Result' to see clinical reference range information.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "EHR Lab Test Analysis | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()