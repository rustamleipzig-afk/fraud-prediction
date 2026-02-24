import pandas as pd
import streamlit as st


class all_reports():

    models = ['LogisticRegression', 'NearestCentroid', 'GaussianNB', 'LabelSpreading',
                'DecisionTreeClassifier', 'QuadraticDiscriminantAnalysis', 'DeepLearningModel']


    reports = {
        'LogisticRegression': {
        'default': {
        '0': {'precision': 0.94, 'recall': 1.00, 'f1-score': 0.97, 'accuracy': 0.94, 'support': 2899},
        '1': {'precision': 0.17, 'recall': 0.01, 'f1-score': 0.01, 'accuracy': 0.94, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.99, 'recall': 0.62, 'f1-score': 0.76, 'accuracy': 0.63, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.90, 'f1-score': 0.23, 'accuracy': 0.63, 'support': 185}
        },
         'boost': {
        '0': {'precision': 0.99, 'recall': 0.61, 'f1-score': 0.76, 'accuracy': 0.62, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.90, 'f1-score': 0.23, 'accuracy': 0.62, 'support': 185}
        }
    },
        'NearestCentroid': {
        'default': {
        '0': {'precision': 0.96, 'recall': 0.65, 'f1-score': 0.78, 'accuracy': 0.65, 'support': 2899},
        '1': {'precision': 0.10, 'recall': 0.61, 'f1-score': 0.17, 'accuracy': 0.65, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.97, 'recall': 0.61, 'f1-score': 0.75, 'accuracy': 0.61, 'support': 2899},
        '1': {'precision': 0.10, 'recall': 0.70, 'f1-score': 0.18, 'accuracy': 0.61, 'support': 185}
        },
         'boost': {
        '0': {'precision': 0.99, 'recall': 0.59, 'f1-score': 0.74, 'accuracy': 0.62, 'support': 2899},
        '1': {'precision': 0.12, 'recall': 0.87, 'f1-score': 0.21, 'accuracy': 0.62, 'support': 185}
        }
    },
        'GaussianNB': {
        'default': {
        '0': {'precision': 0.95, 'recall': 0.91, 'f1-score': 0.93, 'accuracy': 0.87, 'support': 2899},
        '1': {'precision': 0.12, 'recall': 0.18, 'f1-score': 0.14, 'accuracy': 0.87, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.98, 'recall': 0.63, 'f1-score': 0.77, 'accuracy': 0.64, 'support': 2899},
        '1': {'precision': 0.12, 'recall': 0.82, 'f1-score': 0.22, 'accuracy': 0.64, 'support': 185}
        },
         'boost': {
        '0': {'precision': 0.98, 'recall': 0.63, 'f1-score': 0.77, 'accuracy': 0.64, 'support': 2899},
        '1': {'precision': 0.12, 'recall': 0.82, 'f1-score': 0.22, 'accuracy': 0.64, 'support': 185}
        }
    },
        'LabelSpreading': {
        'default': {
        '0': {'precision': 0.95, 'recall': 0.95, 'f1-score': 0.95, 'accuracy': 0.90, 'support': 2899},
        '1': {'precision': 0.16, 'recall': 0.14, 'f1-score': 0.15, 'accuracy': 0.90, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.97, 'recall': 0.58, 'f1-score': 0.72, 'accuracy': 0.59, 'support': 2899},
        '1': {'precision': 0.10, 'recall': 0.76, 'f1-score': 0.18, 'accuracy': 0.59, 'support': 185}
        },
         'boost': {
        '0': {'precision': 0.97, 'recall': 0.57, 'f1-score': 0.72, 'accuracy': 0.64, 'support': 2899},
        '1': {'precision': 0.10, 'recall': 0.74, 'f1-score': 0.19, 'accuracy': 0.64, 'support': 185}
        }
        },
        'DecisionTreeClassifier': {
        'default': {
        '0': {'precision': 0.95, 'recall': 0.93, 'f1-score': 0.94, 'accuracy': 0.88, 'support': 2899},
        '1': {'precision': 0.16, 'recall': 0.23, 'f1-score': 0.19, 'accuracy': 0.88, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.99, 'recall': 0.62, 'f1-score': 0.77, 'accuracy': 0.65, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.92, 'f1-score': 0.23, 'accuracy': 0.65, 'support': 185}
        },
         'boost': {
        '0': {'precision': 1.00, 'recall': 0.58, 'f1-score': 0.73, 'accuracy': 0.60, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.97, 'f1-score': 0.23, 'accuracy': 0.60, 'support': 185}
        }
    },
        'QuadraticDiscriminantAnalysis': {
        'default': {
        '0': {'precision': 0.98, 'recall': 0.92, 'f1-score': 0.93, 'accuracy': 0.87, 'support': 2899},
        '1': {'precision': 0.12, 'recall': 0.18, 'f1-score': 0.15, 'accuracy': 0.87, 'support': 185}
        },
        'hyperparameter': {
        '0': {'precision': 0.98, 'recall': 0.68, 'f1-score': 0.80, 'accuracy': 0.60, 'support': 2899},
        '1': {'precision': 0.14, 'recall': 0.79, 'f1-score': 0.23, 'accuracy': 0.60, 'support': 185}
        },
         'boost': {
        '0': {'precision': 0.98, 'recall': 0.67, 'f1-score': 0.79, 'accuracy': 0.60, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.81, 'f1-score': 0.23, 'accuracy': 0.60, 'support': 185}
        }
    },
    'DeepLearningModel': {
        'default': {
        '0': {'precision': 0.99, 'recall': 0.62, 'f1-score': 0.76, 'accuracy': 0.64, 'support': 2899},
        '1': {'precision': 0.13, 'recall': 0.92, 'f1-score': 0.24, 'accuracy': 0.64, 'support': 185}
        }
    }
    }

    confusion_matrices = {
                            'LogisticRegression':
                            {
                            'default': [[2894, 5], [184, 1]],
                            'hyperparameter': [[1781, 1118], [17, 168]],
                            'boost': [[1813, 1086], [19, 166]]
                            },
                            'DecisionTreeClassifier': 
                            {
                            'default': [[2685, 214], [143, 42]],
                            'hyperparameter': [[1807, 1092], [15, 170]],
                            'boost': [[1676, 1223], [5, 180]]
                            },
                            'NearestCentroid': 
                            {
                            'default': [[1883, 1016], [53, 132]],
                            'hyperparameter': [[1828, 1071], [46, 139]],
                            'boost': [[1697, 1202], [21, 164]]
                            },
                            'QuadraticDiscriminantAnalysis': 
                            {
                            'default': [[2749, 150], [157, 28]],
                            'hyperparameter': [[1961, 938], [38, 147]],
                            'boost': [[1934, 965], [36, 149]]
                            },
                            'GaussianNB': 
                            {
                            'default': [[2646, 253], [152, 33]],
                            'hyperparameter': [[1833, 1066], [33, 152]],
                            'boost': [[1830, 1069], [33, 152]]
                            },
                            'DeepLearningModel': 
                            {
                            'default': [[1801, 1098], [14, 171]]
                            },
                            'LabelSpreading': 
                            {
                            'default': [[1730, 1169], [60, 125]],
                            'hyperparameter': [[1667, 1232], [45, 140]],
                            'boost': [[1767, 1132], [48, 137]]
                            }
                        }
    

    def create_confusion_matrix(self, model, param=None):
        cm = [[0, 0], [0, 0]]  # Dummy confusion matrix
        if param is None:
            cm = self.confusion_matrices[model]['default'] 
        elif 'hyperparameter' in param:
            cm = self.confusion_matrices[model]['hyperparameter'] 
        else:
            cm = self.confusion_matrices[model]['boost'] 

        df_cm = pd.DataFrame(
              cm,
              index=["0", "1"],
              columns=["0", "1"]
              )
        df_cm.index.name = "Actual / Predicted"
        return df_cm

    resampling_reports = {
        'LogisticRegression': {
            'F1 Score': {'SMOTE' : 0.19, 'RandomUnderSampler': 0.19, 'RandomOverSampler': 0.20}},
        'NearestCentroid': {
            'F1 Score': {'SMOTE' : 0.13, 'RandomUnderSampler': 0.14, 'RandomOverSampler': 0.14}},
        'GaussianNB': {
            'F1 Score': {'SMOTE' : 0.21, 'RandomUnderSampler': 0.18, 'RandomOverSampler': 0.19}},
        'DecisionTreeClassifier': {
            'F1 Score': {'SMOTE' : 0.15, 'RandomUnderSampler': 0.17, 'RandomOverSampler': 0.14}},
        'QuadraticDiscriminantAnalysis': {
            'F1 Score': {'SMOTE' : 0.21, 'RandomUnderSampler': 0.14, 'RandomOverSampler': 0.20}},
        'LabelSpreading': {
            'F1 Score': {'SMOTE' : 0.16, 'RandomUnderSampler': 0.16, 'RandomOverSampler': 0.13}}
    }

    def create_resampling_report(self, model):
        if model in self.models:
            resampling_metrics = self.resampling_reports[model]['F1 Score']
            report_df = pd.DataFrame(resampling_metrics, index=['F1 Score'])
            return report_df

    def create_classification_report(self, model, param=None):
        if model in self.models:
            if param is None:
                class_0_metrics = self.reports[model]['default']['0'] 
                class_1_metrics = self.reports[model]['default']['1'] 
            elif 'hyperparameter' in param:
                class_0_metrics = self.reports[f'{model}']['hyperparameter']['0']
                class_1_metrics = self.reports[f'{model}']['hyperparameter']['1']
            else:
                class_0_metrics = self.reports[f'{model}']['boost']['0']
                class_1_metrics = self.reports[f'{model}']['boost']['1']
            report_df = pd.DataFrame({
                '0': class_0_metrics,
                '1': class_1_metrics
            }).transpose()  
            return report_df

    def fill_tables(self, model, param=None):
        st.write(f"Classification report for {model} with {param if param else 'default parameters'}")
        mod_tbl = self.create_classification_report(model, param)
        st.dataframe(mod_tbl, width=500)

        st.write(f"Confusion Matrix for {model} with {param if param else 'default parameters'}")
        df_cm = self.create_confusion_matrix(model, param)
        st.dataframe(df_cm, width=300)

    