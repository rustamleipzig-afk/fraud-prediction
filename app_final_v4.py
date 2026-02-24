import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold                                                   

from sklearn.model_selection import train_test_split

from help_functions import functions
from tables import all_reports as reports
import time                                         
import asyncio

import sys
import streamlit as st
from PIL import Image                     

helper = functions()
rep = reports()

import os

# =====================================================
# Dark Blue Theme + Top Navigation CSS
# =====================================================
custom_css = """
<style>

/* Main background */


.flow-wrap {
    border: 1px solid #2f4a73;
    border-radius: 16px;
    padding: 18px;
    #background: linear-gradient(140deg, rgba(14,22,40,0.95), rgba(26,37,66,0.95));
    margin: 12px 0 14px 0;
}

.flow-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
}

.flow-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}

.flow-box {
    min-width: 240px;
    text-align: center;
    padding: 14px 16px;
    border-radius: 12px;
    color: #ffffff;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

.flow-main {
    background: linear-gradient(120deg, #00b4d8, #0077b6);
}

.flow-left {
    background: linear-gradient(120deg, #ff9f1c, #ff5400);
}

.flow-right {
    background: linear-gradient(120deg, #80ed99, #2a9d8f);
}

.flow-step {
    background: linear-gradient(120deg, #7b2cbf, #4361ee);
    min-width: 180px;
}

.flow-arrow {
    color: #90caf9;
    font-size: 24px;
    font-weight: 800;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
##############################################################################################################################

current_path = os.getcwd()  # aktueller Ordner
parent_path = os.path.dirname(current_path)

print(parent_path)

path_to_data = parent_path + "/fraud-prediction/data"
path_to_images = parent_path + "/fraud-prediction/pic"
path_to_map_data = parent_path + "/fraud-prediction/data/state_fraud_losses.csv"

# =====================================================
# Global Header Image (shown on every page)
# =====================================================
if os.path.exists(path_to_images):
    st.image(path_to_images + "/liora.png", width=130)
else:
    st.info(f"Header image not found at: `{path_to_images}`")

st.markdown(
    "<hr style='border: none; height: 2px; background-color: #7B2CBF; margin: 0 0 1.5rem 0;'>",
    unsafe_allow_html=True
)

@st.cache_resource

def init():
    param_space = {
    'max_depth': (1, 50, 'uniform'),
    'min_samples_split': (2, 20, 'uniform'),
    'min_samples_leaf': (1, 20, 'uniform'),
    'max_features': [None, 'sqrt', 'log2']
    }
    X_train_undersampled = pd.read_csv(f'{path_to_data}/X_train_undersampled.csv', index_col=0)
    y_train_undersampled = pd.read_csv(f'{path_to_data}/y_train_undersampled.csv').squeeze("columns")
    X_test = pd.read_csv(f'{path_to_data}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{path_to_data}/y_test.csv').squeeze("columns")

    # load the selected data:
    X_selected = pd.read_csv(f'{path_to_data}/X_selected.csv', index_col=0)
    y_selected = pd.read_csv(f'{path_to_data}/y_selected.csv').squeeze("columns")
    # split the selected data into train and test sets
    X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42, stratify=y_selected)
    clf_demo = helper.train('DecisionTreeClassifier',X_train_selected, y_train_selected)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bayes_search = BayesSearchCV(estimator=DecisionTreeClassifier(random_state=42), 
                            search_spaces=param_space, cv=cv, scoring='balanced_accuracy', n_jobs=-1, n_iter=20, random_state=42)
    bayes_search.fit(X_train_undersampled, y_train_undersampled)

    model = BaggingClassifier(DecisionTreeClassifier(**bayes_search.best_params_), n_estimators=100, random_state=42)
                                                                          
    model.fit(X_train_undersampled, y_train_undersampled)
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr = pd.DataFrame(cr).transpose().round(2)                          
    accuracy_value = cr.loc["accuracy", "f1-score"]
    cr.insert(cr.columns.get_loc("support"), "accuracy", "")
    cr.loc["0", "accuracy"] = accuracy_value
    cr.loc["1", "accuracy"] = accuracy_value
    cr = cr.iloc[:-3]
    cm = pd.crosstab(y_test, y_pred, margins=True, rownames=['Actual'], colnames=['Predicted'])
    cm.index.name = "Actual / Predicted"
    cm = cm.drop(index="All", errors="ignore")
    cm = cm.drop(columns="All", errors="ignore")
    cm = pd.DataFrame(cm)

    return cr, cm, X_train_undersampled, y_train_undersampled, X_test, y_test, clf_demo, X_test_selected, y_test_selected, model
c_report, c_matrix, X_train_undersampled, y_train_undersampled, X_test, y_test, clf_demo, X_test_selected, y_test_selected, model = init()

def reset_checkbox():
    st.session_state['checkbox_checked'] = False
# --- PAGE CONFIGURATION ---
# App Configuration
# =====================================================
st.set_page_config(
    page_title="Fraud Detection Project",
    layout="wide"
)
 # =====================================================
 
# --- CUSTOM STYLING ---
#st.title("🛡️ Insurance Claims Dashboard")
#st.markdown("Analyze insurance claim patterns and fraud reports in real-time.")
 
# --- SIDEBAR: DATA INPUT ---
st.sidebar.header("Data Control Center")
uploaded_file = st.sidebar.file_uploader("Upload your fraud_oracle.csv", type=["csv"])

#st.title("Fraud prediction : binary classification project")
st.sidebar.title("Table of contents")
pages=["Cover page", "Introduction", "DataVizualization", "Modelling",  "Demo-Live", 'Interpretation and conclusion']
page=st.sidebar.radio("Go to", pages)

if uploaded_file is not None:
  df=pd.read_csv(uploaded_file, sep=';')

  if page == pages[0] :
     
     # Vertical spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Center container using columns
    left, center, right = st.columns([1, 3, 1])

    with center:
        st.markdown(
            "<h1 class='page-title'>Predictive Analysis of Fraudulent Insurance Claims</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<h3 class='cover-center'>Rustam Bershakhov · Adnane Fria · Ahmad Melhem · Mangesh Pise</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='cover-center'><strong>Type:</strong> Data Science Project Report<br><strong>Presented on:</strong> February 24, 2026</p>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown("<h3 class='cover-center'>Abstract</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p class='cover-abstract'>Insurance fraud poses a major challenge for insurance companies, resulting in significant financial losses and increased premiums for genuine policyholders. With the growing availability of historical claim data, machine learning techniques offer effective tools for identifying fraudulent patterns and improving decision-making.</p>",
            unsafe_allow_html=True
        )
  
  if page == pages[1] :
    #st.header("1. Introduction")
    st.markdown("<h2 class='cover-left'>Introduction</h2>", unsafe_allow_html=True)

    st.subheader("Background, Motivation, and Detection Approach")
    st.markdown(
        """
        <div class="intro-bullets">
            <ul>
                <li>Insurance fraud creates substantial financial losses and raises premiums for honest policyholders.</li>
                <li>Growing claim volumes make manual review slower, costlier, and more error-prone.</li>
                <li>Traditional rule-based detection is hard to scale and struggles with evolving fraud behavior.</li>
                <li>Machine learning improves detection by learning complex, non-linear risk patterns from historical claims.</li>
                <li>A data-driven workflow supports faster and more consistent fraud screening decisions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("U.S. Fraud Reports and Losses by State (2024)")

    if not os.path.exists(path_to_map_data):
        st.warning(f"Map data file not found: `{path_to_map_data}`")
    else:
        try:
            df_map = helper.load_state_map_data(path_to_map_data)

            metric = st.radio(
                "Map variable",
                ["fraud_cases", "losses_usd"],
                horizontal=True,
                format_func=lambda x: "Fraud Cases" if x == "fraud_cases" else "Total Losses (USD)"
            )

            colorbar_title = "Fraud Cases" if metric == "fraud_cases" else "Losses (USD)"

            fig = px.choropleth(
                df_map,
                locations="state",
                locationmode="USA-states",
                color=metric,
                scope="usa",
                color_continuous_scale=[
                    [0.0, "#0B132B"],
                    [0.25, "#1C2541"],
                    [0.5, "#2D3E66"],
                    [0.75, "#3A506B"],
                    [1.0, "#5BC0BE"],
                ],
                hover_data={
                    "state": True,
                    "fraud_cases": True,
                    "losses_usd": ":,.2f",
                },
                labels={
                    "fraud_cases": "Fraud Cases",
                    "losses_usd": "Losses (USD)",
                    "state": "State",
                },
            )

            fig.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAF2FF"),
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    lakecolor="#0D1B2A",
                    landcolor="#112240",
                    subunitcolor="#415A77",
                ),
            )
            fig.update_coloraxes(
                colorbar_title=colorbar_title,
                colorbar=dict(
                    tickfont=dict(color="#EAF2FF"),
                    title=dict(text=colorbar_title, font=dict(color="#EAF2FF")),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Source: Federal Trade Commission (FTC), Consumer Sentinel Network Annual Data Book 2024.")

            total_cases = int(df_map["fraud_cases"].sum())
            total_losses = float(df_map["losses_usd"].sum())
            totals_df = pd.DataFrame(
                [
                    {
                        "Year": 2024,
                        "Total Reported Fraud Cases": total_cases,
                        "Total Losses (USD)": total_losses,
                    }
                ]
            )
            styled_totals = (
                totals_df.style
                .format({"Total Losses (USD)": "${:,.2f}"})
                .set_properties(**{"text-align": "center", "font-weight": "700", "color": "#f8fbff"})
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#0b132b"), ("color", "#f8fbff"), ("text-align", "center"), ("font-size", "15px")]},
                    {"selector": "td:nth-child(1)", "props": [("background-color", "#0077b6")]},
                    {"selector": "td:nth-child(2)", "props": [("background-color", "#ff7b00")]},
                    {"selector": "td:nth-child(3)", "props": [("background-color", "#2a9d8f")]},
                ])
            )
            st.dataframe(styled_totals, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Unable to render map from `{path_to_map_data}`: {e}")

    st.subheader("Dataset")
    st.markdown(
        """
        <div class="flow-wrap">
            <div class="flow-col">
                <div class="flow-box flow-main">fraud_oracle.csv</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-row">
                    <div class="flow-box flow-left">Total Rows<br>15,420</div>
                    <div class="flow-box flow-right">Total Columns<br>33</div>
                </div>
                <div class="flow-arrow">↓</div>
                <div class="flow-row">
                    <div class="flow-box flow-left">Features<br>32 (Mostly Categorical)</div>
                    <div class="flow-box flow-right">Target Variable<br>FraudFound_P</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Objectives and Workflow")
    st.markdown(
        """
        <div class="flow-wrap">
            <div class="flow-col">
                <div class="flow-box flow-main">Main Objective: Build a model that can detect fraudulent insurance claims.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-wrap">
            <div class="flow-col">
                <div class="flow-box flow-step">1. EDA & Data Understanding</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-box flow-step">2. Preprocessing</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-box flow-step">3. Model Training & Optimazation</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-box flow-step">4. Interpretation & Feature Importance</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-box flow-step">5. Evaluation & Conclusions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  if page == pages[2] : 
    st.write("### DataVizualization")

    # select per features and per target variable and show the distribution of the target variable for each feature variable 
    for ele in ['FraudFound_P', 'Age', 'VehicleCategory', 'AccidentArea', 'VehiclePrice', 'AgeOfVehicle', 'BasePolicy', 'NumberOfSuppliments', "Fault"]:#, 'PastNumberOfClaims', 'AddressChange_Claim', 'Fault', 'Make', 'VehicleCategory', 'VehiclePrice', 'AgeOfVehicle', 'AccidentArea', 'BasePolicy', 'PolicyType', 'Deductible', 'NumberOfSuppliments', 'AgentType', 'PoliceReportFiled', 'WitnessPresent', 'Days_Policy_Claim', 'Days_Policy_Accident', 'WeekOfMonthClaimed', 'MonthClaimed', 'Month', 'DayOfWeekClaimed', 'DayOfWeek', 'Year']:

      if ele == 'FraudFound_P':
        st.write("Distribution of the target variable")
        fig = px.pie(df, names='FraudFound_P', title=f'Pie Chart of FraudFound_P')
        st.plotly_chart(fig)
        st.markdown("""
                    <div style='padding: 12px; border-radius: 8px; 
                                background-color: #e3f2fd; 
                                border: 1px solid #90caf9; 
                                margin-top: 15px;'>
                        <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                            Approximately <strong>94%</strong> of the observations correspond to non‑fraud cases, 
                            whereas only about <strong>6%</strong> represent fraud cases, 
                            indicating a strong class imbalance.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
      
      elif ele == 'Age':
        fig = px.box(df, x='FraudFound_P', y=ele, title=f'Boxplot of {ele} by FraudFound_P')
        st.plotly_chart(fig)
        st.markdown("""
                    <div style='padding: 12px; border-radius: 8px; 
                                background-color: #e3f2fd; 
                                border: 1px solid #90caf9; 
                                margin-top: 15px;'>
                        <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                            For the 'Age' variable, the minimum value is 0 and the maximum is 80. The value 0 is clearly outliers and need to be processed.
                            The boxplot for fraud cases shows a concentration of individuals between 30 and 46 years old, suggesting a potential relationship between this age range and reported fraud. 
                    This association is supported by the ANOVA test, which yields a p‑value below 0.05.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        st.write("")
      else:
        fig = px.histogram(df, x=ele, color='FraudFound_P', histnorm='percent', barmode='group', title=f'Distribution of {ele} by FraudFound_P')
        st.plotly_chart(fig)
        if ele == 'VehicleCategory':
          st.markdown("""
                    <div style='padding: 12px; border-radius: 8px; 
                                background-color: #e3f2fd; 
                                border: 1px solid #90caf9; 
                                margin-top: 15px;'>
                        <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                            The dataset is dominated by the ‘Sedan’ category, which accounts for 63% of all vehicles, followed by ‘Sport’ vehicles at 35% and ‘Utility’ vehicles at just 2%. 
                            This imbalance is reflected in the fraud distribution, where 86% of all detected fraud cases are linked to the ‘Sedan’ category
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
          st.write("")

        elif ele == 'AccidentArea':
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'>
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                          Approximately 90% of the accidents in the dataset take place in urban areas, while only 10% occur in rural regions. This imbalance is reflected in the fraud distribution: 
                          86% of detected fraud cases originate from urban accident locations, indicating a significantly higher fraud presence in these areas.
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")
        
        elif ele == "VehiclePrice":
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'> 
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                              The distribution shows that vehicles priced between 20,000 and 29,000 represent half of the dataset (52%), followed by vehicles priced between 30,000 and 39,000 at 23%. High‑value vehicles above 69,000 account for 14%. 
                              The histogram reflects the same pattern: more than 64% of fraud cases involve vehicles priced between 20,000 and 39,000, and around 20% of fraud cases are linked to vehicles valued above 69,000.
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")
          
        elif ele == "AgeOfVehicle":
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'>
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                              The data indicates that vehicles older than six years constitute the predominant category among the distribution in the dataset, 
                              it reflects the same in the reported fraud cases (82%).
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")
        
        elif ele == "BasePolicy":
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'> 
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                              The BasePolicy variable consists of three categories, whose proportions in the dataset are relatively similar: 38% Collision, 32% Liability, and 29% All Perils. 
                              In contrast, the fraud distribution reveals a strong concentration in the "All Perils" (49%) and "Collision" (47%) categories, indicating that these policy types are more frequently associated with detected fraud cases.
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")

        elif ele == "NumberOfSuppliments":
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'> 
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                              The NumberOfSupplements variable reflects how many supplementary claims were filed, representing additional repair costs beyond the initial claim. The distribution ranges from 0 (none) supplements, which is the most common category (45%), to "more than 5" supplements (25%). 
                              The fraud distribution shows a similar pattern, with the majority of fraud cases occurring in the "0 supplements" category with 51% the rest represent claims with at least 1 supplementary claims (49%)
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")

        elif ele == "Fault":
          st.markdown("""
                  <div style='padding: 12px; border-radius: 8px; 
                              background-color: #e3f2fd; 
                              border: 1px solid #90caf9; 
                              margin-top: 15px;'> 
                      <p style='font-size: 15px; line-height: 1.5; color:#0d47a1; margin: 0;'>
                              The data indicates that 73% of accidents were attributed to the policyholder, while only 27% were caused by third parties. This substantial imbalance is mirrored in the fraud distribution, where the majority of fraud cases originate from policyholders. 
                              Given their significantly higher share of fault, this relationship appears consistent and reasonable.
                      </p>
                  </div>
                  """, unsafe_allow_html=True)
          st.write("")
      
      if ele == 'Age':
         if st.checkbox("Show statistical test results for Age", key='age_' + ele):
          result = helper.check_relationship_between_quantitative_and_qualitative_variables(df, ele, 'FraudFound_P')
          
          st.write("Group statistics:")
          st.dataframe(result)
      elif ele == 'FraudFound_P':
        pass

      else:
          if st.checkbox("Show Contingency table", key='chi2_' + ele):
            result = helper.check_relationship_between_2_qualitative_variables(df, ele, 'FraudFound_P')
            #st.write("Chi-squared test results for " + ele + ":")
            #st.write(f"Chi-squared statistic: {result['chi2_statistic']}")
            #st.write(f"P-value: {result['p_value']}")
            #st.write(f"Degrees of freedom: {result['degrees_of_freedom']}")
            st.write("Contingency table:")
            st.dataframe(result['contingency_table'])
            

            if st.checkbox("Show Chi2 test and cramerV value for each category of the feature variable", key='cramer_' + ele):
              l_Category = []
              l_cramer = []
              l_p_value = []
              for i in pd.get_dummies(df[ele]):
                cramer_v, p_value = helper.check_qualitative_correlation_with_chi2_and_Vcramer(df, ele, 'FraudFound_P', category=i)

                if p_value < 0.05:
                    #st.write(f"{i} : Cramér's V: {cramer_v}, p-value: {p_value}")
                    l_Category.append(i)
                    l_cramer.append(cramer_v)
                    l_p_value.append(round(p_value, 4))
                    
                    
              df_cramer = pd.DataFrame({
                  "Category": l_Category,
                  "Cramér's V": l_cramer,
                  "p-value": l_p_value

              })
              # überptüfen ob df_cramer leer ist:
              if df_cramer.empty:
                  st.write("No significant association found for any category.")
              else:
                st.dataframe(df_cramer)



if page == pages[3] : 
 #st.write("### Modelling")
  #cm = None                               

  st.write("### Results of LazyPredict for the top models")

  lazy_res = pd.DataFrame([
    ["NearestCentroid",                0.67, 0.70, 0.70, 0.76],
    ["DecisionTreeClassifier",         0.89, 0.58, 0.58, 0.90],
    ["LabelSpreading",                 0.90, 0.56, 0.56, 0.90],
    ["LabelPropagation",               0.89, 0.56, 0.56, 0.90],
    ["QuadraticDiscriminantAnalysis",  0.87, 0.55, 0.55, 0.88],
    ["Perceptron",                     0.87, 0.55, 0.55, 0.88],
    ["GaussianNB",                     0.87, 0.54, 0.54, 0.88],
    ["BaggingClassifier",              0.93, 0.53, 0.53, 0.92],
    ["ExtraTreeClassifier",            0.89, 0.53, 0.53, 0.89],
    ["XGBClassifier",                  0.94, 0.53, 0.53, 0.92],
], columns=["Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score"])


  lazy_sorted = lazy_res.sort_values(by="Balanced Accuracy", ascending=False)
                            
           

  st.dataframe(lazy_sorted, width='stretch', height='auto')

  # -----------------------------
  # UI: Model Buttons
  # -----------------------------
  st.write("# ML models comparison")
  models_mod = ['LogisticRegression', 'NearestCentroid', 'LabelSpreading',
                'DecisionTreeClassifier', 'QuadraticDiscriminantAnalysis']
  
  
  cols = st.columns(len(models_mod))
  
  for col, model in zip(cols, models_mod):
      with col:
        btn = st.button(model, width='stretch', type='primary')
         
      if btn:

        st.subheader(f"Choose {model} variant", width='stretch')
        tab_def, tab_resample, tab_hyperparameter, tab_boost = st.tabs(["Default", "Resampling","Hyperparameter Tuning", "Boosting"])
                                 
                                         

        with tab_def:
            rep.fill_tables(model)  # Fill tables for the default parameters
        
        with tab_resample:
            st.write(f"Results for {model} with resampling")
            resample_tbl = rep.create_resampling_report(model)
            st.dataframe(resample_tbl, width=500)
            
        with tab_hyperparameter:
            rep.fill_tables(model, param='hyperparameter tuning')  # Fill tables for the hyperparameter tuning

        with tab_boost:
            rep.fill_tables(model, param='BaggingClassifier')  # Fill tables for the boosting   

  # plot graph with the f1-score of each model for the default parameters, hyperparameter tuning and boosting
  if st.checkbox("Show models performance comparison") :
    st.write("# Models performance comparison")
    f1_scores = []
    for model in models_mod:
      f1_default = rep.reports[model]['default']['1']['f1-score']
      f1_hyper = rep.reports[model]['hyperparameter']['1']['f1-score']
      f1_boost = rep.reports[model]['boost']['1']['f1-score']
      f1_scores.append({
          "Model": model,
          "Default": f1_default,
          "Hyperparameter Tuning": f1_hyper,
          "Boosting": f1_boost
      })
    f1_df = pd.DataFrame(f1_scores)
    f1_df_melted = f1_df.melt(id_vars='Model', var_name='Variant', value_name='F1 Score')
    fig = px.bar(f1_df_melted, x='Model', y='F1 Score', color='Variant', barmode='group', title='F1 Score Comparison of Models and Variants')
    st.plotly_chart(fig)

    # get the best f1-score for each model and choose the best model based on the f1-score and recall
    best_models = []
    for model in models_mod:
                                                             
      f1_default = rep.reports[model]['default']['1']['f1-score']
      f1_hyper = rep.reports[model]['hyperparameter']['1']['f1-score']
      f1_boost = rep.reports[model]['boost']['1']['f1-score']
      best_f1 = max(f1_default, f1_hyper, f1_boost)
      if best_f1 == f1_default:
        variant = 'Default'
      elif best_f1 == f1_hyper:
        variant = 'Hyperparameter Tuning'
      else:
        variant = 'Boosting'
      best_models.append({
          "Model": model,
          "Best F1 Score": best_f1,
          'Best Recall': max(rep.reports[model]['default']['1']['recall'], rep.reports[model]['hyperparameter']['1']['recall'], rep.reports[model]['boost']['1']['recall']),
          "Variant": variant
      })
    best_models_df = pd.DataFrame(best_models)
    best_models_df = best_models_df.sort_values(by=['Best F1 Score', 'Best Recall'], ascending=False)
    best_models_df = best_models_df.reset_index(drop=True)
    st.write("### Best models based on F1 Score and Recall")
    st.dataframe(best_models_df, width=600)
   
  ### Section for feature selection results
  if st.checkbox("Show feature selection results with smf logit model") :
    st.write("### Feature selection with smf logit model")                                                        

    df = pd.DataFrame([
    ["Fault",                   2.6443,   14.0743,   0.0],
    ["BasePolicy",              0.8026,    2.2314,   0.0],
    ["VehiclePrice",            0.1222,    1.13,     0.0],
    ["NumberOfSuppliments",    -0.0672,    0.935,    0.0272],
    ["AccidentArea",           -0.2481,    0.7803,   0.0158],
    ["MonthClaimed_cos",       -0.2665,    0.7661,   0.0342],
    ["Sex",                    -0.2703,    0.7631,   0.0163],
    ["VehicleCategory_Utility",-0.6,       0.5488,   0.0015],
    ["VehicleCategory_Sport",  -0.9107,    0.4022,   0.0],
    ["Age",                    -0.0143,    0.9822,   0.0014],
    ["Deductible_700",         -2.0404,    0.13,     0.0382],
    ["Deductible_400",         -2.1595,    0.1154,   0.0231],
    ], columns=["Feature", "Coefficient", "Odds Ratio", "P-Value"])

    row_height = 35  # adjust if needed
    height = (len(df) + 1) * row_height
    st.dataframe(df, width='stretch', height=height)

    best_after_feature_selection = pd.DataFrame([
    ["DecisionTreeClassifier",         0.23, 0.97, "Boosting"],
    ["LogisticRegression",             0.23, 0.92, "Boosting"],
    ["QuadraticDiscriminantAnalysis",  0.23, 0.97, "Boosting"],
    ["NearestCentroid",                0.22, 0.85, "Boosting"],
    ["LabelSpreading",                 0.21, 0.95, "Boosting"],
    ], columns=["Model", "Best F1 Score", "Best Recall", "Variant"]).sort_values(
    by=["Best F1 Score", "Best Recall"],
    ascending=[False, False]
    ).reset_index(drop=True)
    st.write("### Best models after feature selection based on F1 Score and Recall")
    row_height = 35  # adjust if needed
    height = (len(best_after_feature_selection) + 1) * row_height
    st.dataframe(best_after_feature_selection, width=600, height=height)




  ### Section for Deep Learning model results
  st.write("# Deep learning model")
  dl_btn = st.button("DEEP LEARNING", type='primary')
  if dl_btn:
    #st.write(f"Best results for Deep Learning model")                                                           

    col1, col2 = st.columns([1, 2])                     

    with col1:
      st.image(f"{path_to_images}/dl.png", caption="Deep Learning model architecture", width='stretch')                        

    with col2:
      mod_tbl = rep.create_classification_report('DeepLearningModel')
      st.dataframe(mod_tbl, width=500)

      #st.write(f"Confusion Matrix for Deep Learning model with default parameters")
      df_cm = rep.create_confusion_matrix('DeepLearningModel')
      st.dataframe(df_cm, width=300)

if page == pages[4] :
    st.write("### Demo-Live prediction")
    st.write("### Best model: BaggingClassifier with DecisionTreeClassifier as base estimator")                  

    if st.checkbox("Show the selected claim for the demo prediction"):
        if st.slider("Age of Policyholder", 18, 80, 30, key='age'):
            #st.write("Selected Age: ", st.session_state['age'])
            X_test_selected.loc[X_test_selected.index[16], 'Age'] = st.session_state['age']
            #use minmax scaler to scale the age value between 0 and 1 based on the min and max age in the training data
            #from sklearn.preprocessing import MinMaxScaler
            #scaler = MinMaxScaler()
            #X_test_selected.loc[X_test_selected.index[16], 'Age'] = scaler.fit_transform(X_test_selected.loc[X_test_selected.index[16], 'Age'])
            X_test_selected.loc[X_test_selected.index[16], 'Age'] = (X_test_selected.loc[X_test_selected.index[16], 'Age'] - 18)/(80-18)


        
        if st.selectbox("Vehicle Category", ["Sport", "Utility"], key="Sedan") == "Sport":
            #.write("Selected Vehicle Category: Sport")
            X_test_selected.loc[X_test_selected.index[16], 'VehicleCategory_Sport'] = 1
            X_test_selected.loc[X_test_selected.index[16], 'VehicleCategory_Utility'] = 0
        else:
            #st.write("Selected Vehicle Category: Utility")
            X_test_selected.loc[X_test_selected.index[16], 'VehicleCategory_Sport'] = 0
            X_test_selected.loc[X_test_selected.index[16], 'VehicleCategory_Utility'] = 1

        if st.selectbox("Fault", ["Policy Holder", "Third Party"], key='Urban') == "Policy Holder":
            #st.write("Selected Fault: Third Party")
            X_test_selected.loc[X_test_selected.index[16], 'Fault'] = 1
        else:
            #st.write("Selected Fault: PolicyHolder")
            X_test_selected.loc[X_test_selected.index[16], 'Fault'] = 0

        if st.slider("Vehicle Price", 10000, 70000, 30000, key='vehicle_price'):
            
            vehicle_price_mapping = {'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2, '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5}
            #X['VehiclePrice'] = X['VehiclePrice'].map(vehicle_price_mapping)
            if st.session_state['vehicle_price'] < 20000:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 0
            elif 20000 < st.session_state['vehicle_price'] < 30000:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 1
            elif 30000 < st.session_state['vehicle_price'] < 40000:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 2
            elif 40000 < st.session_state['vehicle_price'] < 60000:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 3
            elif 60000 < st.session_state['vehicle_price'] < 70000:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 4
            else:
               X_test_selected.loc[X_test_selected.index[16], 'VehiclePrice'] = 5
        
        if st.selectbox("Sex", ["Male", "Female"], key='sex') == "Male":
            #st.write("Selected Sex: Male")
            X_test_selected.loc[X_test_selected.index[16], 'Sex'] = 0
        else:
            #st.write("Selected Sex: Female")
            X_test_selected.loc[X_test_selected.index[16], 'Sex'] = 1

        base_policy = st.selectbox("Base Policy", ["Collision", "Liability", "All Perils"], key="base_policy")
        for i in ["Collision", "Liability", "All Perils"]:
            if base_policy == i:
                #st.write(f"Selected Base Policy: {i}")
                if i == "All Perils":
                    X_test_selected.loc[X_test_selected.index[16], 'BasePolicy'] = 3
                elif i == "Collision":
                    X_test_selected.loc[X_test_selected.index[16], 'BasePolicy'] = 2
                else:
                    X_test_selected.loc[X_test_selected.index[16], 'BasePolicy'] = 1
                break
    
    #st.dataframe(X_test_selected.iloc[[16]])
    #st.dataframe(y_test_selected.head(10))
    st.write("### Prediction results for the selected claim")
    #if st.checkbox("Show prediction for the selected claim"):
    scores = helper.scores(model, X_test_selected.iloc[[16]], y_test_selected.iloc[[16]], output_dict=True)
    prediction = model.predict(X_test_selected.iloc[[16]])
    
    score = prediction
 
    color = "#4CAF50" if score == 0 else "#F44336"
    
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; font-weight:bold;'>"
        f"Fraud detected: {score.item()} (1 = Fraud, 0 = Not Fraud)"
        "</div>",
        unsafe_allow_html=True
    )

  ##############
if page == pages[5]:
    st.write("### Interpretation and conclusion")

    # -----------------------------
    # 1. Session State vorbereiten
    # -----------------------------
    if "clf" not in st.session_state:
        st.session_state.clf = None

    # -----------------------------
    # 2. Button: Modell einmal trainieren
    # -----------------------------
    if st.button("Show Predicted Probabilities", type='primary'):
        clf = LogisticRegression(class_weight='balanced')
        clf.fit(X_train_undersampled, y_train_undersampled)
        st.session_state.clf = clf   # Modell speichern

    # Wenn noch kein Modell existiert → Hinweis + Abbruch
    if st.session_state.clf is None:
        #st.info("Bitte zuerst auf **Show Predicted Probabilities** klicken.")
        st.stop()

    # Modell laden
    clf = st.session_state.clf

    # -----------------------------
    # 3. Threshold Slider (immer sichtbar)
    # -----------------------------
    st.markdown("""
    ### Adjusting the classification threshold  
    Use the slider below to adjust the classification threshold and see how it affects the distribution of predicted probabilities for False Positives, True Positives, and False Negatives.
    """)


    # -----------------------------
    # 4. Neue Predictions basierend auf Threshold
    # -----------------------------
    y_proba_all, y_proba = helper.calculate_probabilities(clf, X_test)
    y_pred = clf.predict(X_test)

    #fig = px.histogram(y_proba, x=1, color=0, nbins=50, title=f"Distribution of Predicted Probabilities for {clf.__class__.__name__}")
    #st.plotly_chart(fig, use_container_width=True)

    probe = pd.DataFrame(y_proba_all)
    probe['prediction'] = y_pred
    probe['real value'] = y_test.values

    if st.checkbox("show thresholded predictions", key="thresholded_predictions"):
      threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
      probe = pd.DataFrame(y_proba_all)
      probe['prediction'] = (probe[1] >= threshold).astype(int)
      probe['real value'] = y_test.values

      # -----------------------------
      # 5. Confusion Matrix anzeigen
      # -----------------------------
      from sklearn.metrics import confusion_matrix
      cm = confusion_matrix(y_test, probe['prediction'])
      st.subheader("Confusion Matrix")
      st.dataframe(cm)
      
    FP = probe[(probe['prediction'] == 1) & (probe['real value'] == 0)]
    TP = probe[(probe['prediction'] == 1) & (probe['real value'] == 1)]
    FN = probe[(probe['prediction'] == 0) & (probe['real value'] == 1)]
    TN = probe[(probe['prediction'] == 0) & (probe['real value'] == 0)]

    #st.dataframe(TP.head())

    # -----------------------------
    # 6. Boxplots nebeneinander + Farben + Linien
    # -----------------------------
    col1, col_line1, col2, col_line2, col3 = st.columns([3, 0.2, 3, 0.2, 3])

    with col1:
        fig_fp = px.box(FP, x=1, title='Type 1 Error: False Positives', color_discrete_sequence=["#1f77b4"])
        fig_fp.update_layout(height=400)
        st.plotly_chart(fig_fp, use_container_width=True)

    with col_line1:
        st.markdown("<div style='border-left: 2px solid #ccc; height: 400px; margin: auto;'></div>",
                    unsafe_allow_html=True)

    with col2:
        fig_fn = px.box(FN, x=1, title='Type 2 Error: False Negatives', color_discrete_sequence=["#d62728"])
        fig_fn.update_layout(height=400)
        st.plotly_chart(fig_fn, use_container_width=True)

    with col_line2:
        st.markdown("<div style='border-left: 2px solid #ccc; height: 400px; margin: auto;'></div>",
                    unsafe_allow_html=True)

    with col3:
        fig_tp = px.box(TP, x=1, title='True Positives', color_discrete_sequence=["#2ca02c"])
        fig_tp.update_layout(height=400)
        st.plotly_chart(fig_tp, use_container_width=True)
    
    st.markdown("""
<div style='padding: 12px; border-radius: 8px; border: 1px solid #444; margin-bottom: 20px;'>

<h4 style='margin-top: 0;'>🧠 Interpretation of Predicted Probability Distributions</h4>

<p style='font-size: 15px; line-height: 1.5;'>
    These boxplots summarize how the classifier assigns fraud probabilities across different prediction outcomes.
    <strong style='color:#1f77b4;'>False Positives</strong> represent legitimate claims that received high predicted fraud scores, indicating overestimation.
    <strong style='color:#2ca02c;'>True Positives</strong> show consistently high predicted probabilities, reflecting correctly identified fraudulent cases.
    <strong style='color:#d62728;'>False Negatives</strong> correspond to fraudulent claims that were assigned low fraud probabilities, revealing systematic errors.
    Together, these distributions illustrate how the decision threshold shapes model sensitivity and highlight regions where misclassification is most likely.
</p>

</div>
""", unsafe_allow_html=True)

      
    # add a line to seperate the section of the predicted probabilities from the rest of the page:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Analysis of the shap values of the best model")
    # add new button to show the shap values of the best model:
    if st.button("Show SHAP values", type='primary'): 
      # kannst du das Bild zentrieren und etwas größer machen, damit man die SHAP-Werte besser sehen kann?
      col_left, col_center, col_right = st.columns([1, 2, 1])

      with col_center:
          st.image("shape_streamlit.png", caption="Shape values of the selected model")

      # kannst du diese Punkteliste in einem schön formatierten Textblock anzeigen, damit die wichtigsten Erkenntnisse aus der Analyse der SHAP-Werte übersichtlich dargestellt werden?
      st.markdown("""
<div style='padding: 15px; border-radius: 10px; border: 1px solid #444;'>

<h3>🔍 Key Insights from Feature Influence</h3>

<ul style='font-size:16px; line-height:1.6;'>

<li><strong>Most influential feature:</strong><br>
<span style='color:#c0392b;'>Fault (Policyholder at fault)</span> → strongest driver increasing fraud probability.</li>

<li><strong>Other strong fraud indicators:</strong><br>
<span style='color:#8e44ad;'>BasePolicy (All Perils)</span> → increases fraud risk.<br>
<span style='color:#8e44ad;'>Deductible_400</span> → associated with higher fraud suspicion.</li>

<li><strong>Protective (fraud‑reducing) features:</strong><br>
<span style='color:#27ae60;'>Deductible_700</span> → strongly shifts predictions toward legitimate claims.<br>
<span style='color:#27ae60;'>VehicleCategory_Sport & Utility</span> → lower fraud likelihood.</li>

<li><strong>Overall insight:</strong><br>
Fraud predictions are mainly driven by fault attribution, policy type, and deductible level, while vehicle category plays a secondary protective role.</li>

</ul>

</div>
""", unsafe_allow_html=True)
      
      st.markdown("<hr>", unsafe_allow_html=True)
      st.write("### Conclusion")
      st.markdown("""
    - This project was a comprehensive journey through handling severe class imbalance in real-world data.
    - Standard modeling and simple resampling are not enough when the minority class mimics the majority class closely.
    - Ensemble techniques like Bagging, class-weighted Logistic Regression, and customized Deep Learning architectures enabled highly sensitive models that detect over 90% of fraudulent claims.
    - Due to inherent overlap in claim features, false positives are unavoidable when aiming to capture nearly all fraud.
    - The most practical real-world deployment is an automated, high-recall screening tool.
    - The system flags the suspicious 10-15% of claims for human investigators, saving substantial time and cost while protecting honest policyholders.
    """)
      
