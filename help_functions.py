import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

class functions():

    def get_model_instance(self, model_name):
        """
        Retrieve hyperparameter grid for a given machine learning model.
        This method returns a dictionary of hyperparameters and their possible values
        for use with scikit-learn's GridSearchCV or similar hyperparameter tuning tools.
        Parameters
        ----------
        model_name : str
            The name of the classifier model. Supported models include:
            - BaggingClassifier
            - LGBMClassifier
            - XGBClassifier
            - SVC
            - KNeighborsClassifier
            - RandomForestClassifier
            - BernoulliNB
            - ExtraTreesClassifier
            - RidgeClassifier
            - LinearDiscriminantAnalysis
            - RidgeClassifierCV
            - SGDClassifier
            - LogisticRegression
            - DummyClassifier
            - CalibratedClassifierCV
            - LinearSVC
            - AdaBoostClassifier
            - PassiveAggressiveClassifier
            - LabelSpreading
            - LabelPropagation
            - ExtraTreeClassifier
            - DecisionTreeClassifier
            - QuadraticDiscriminantAnalysis
            - Perceptron
            - GaussianNB
            - NearestCentroid
        Returns
        -------
        dict
            A dictionary where keys are hyperparameter names and values are lists
            of possible values to test during grid search.
        Raises
        ------
        KeyError
            If the provided model_name is not in the param_grids dictionary.
        Examples
        --------
        >>> param_grid = get_param_grid("RandomForestClassifier")
        >>> param_grid["n_estimators"]
        [100, 300, 500]
        """

        param_grids = {
            "BaggingClassifier": BaggingClassifier,
            "LGBMClassifier": LGBMClassifier,
            "XGBClassifier": XGBClassifier,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "BernoulliNB": BernoulliNB,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "RidgeClassifier": RidgeClassifier,
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
            "RidgeClassifierCV": RidgeClassifierCV,
            "SGDClassifier": SGDClassifier,
            "LogisticRegression": LogisticRegression,
            "DummyClassifier": DummyClassifier,
            "CalibratedClassifierCV": CalibratedClassifierCV,
            "LinearSVC": LinearSVC,
            "AdaBoostClassifier": AdaBoostClassifier,
            "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
            "LabelSpreading": LabelSpreading,
            "LabelPropagation": LabelPropagation,
            "ExtraTreeClassifier": ExtraTreeClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
            "Perceptron": Perceptron,
            "GaussianNB": GaussianNB,
            "NearestCentroid": NearestCentroid
        }

        return param_grids[model_name]

    def encode_using_onehotencoder(self, X_train, X_test, lCategorical_cols):
        """
        Encode categorical variables in a DataFrame using OneHotEncoder.
        This method converts all categorical columns in the provided DataFrame
        into numerical format using sklearn's OneHotEncoder.
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing categorical variables to be encoded.
        lCategorical_cols : list of str
            List of column names to be encoded.
        Returns
        -------
        pd.DataFrame
            A new DataFrame with categorical variables encoded as numerical variables.
        Notes
        -----
        - Non-categorical columns are not modified.
        - The original DataFrame is not altered; a new DataFrame is returned.
        Examples
        --------
        >>> df = pd.DataFrame({'A': ['cat', 'dog', 'cat'], 'B': [1, 2, 3]})
        >>> encoded_df = encode_categorical_variables_using_onehotencoder(df)
        """
        from sklearn.preprocessing import OneHotEncoder

        #categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
        
        if X_train is not None:
            # Fit + transform auf Trainingsdaten
            encoded_data_X_train = encoder.fit_transform(X_train[lCategorical_cols])
            encoded_X_train = pd.DataFrame(encoded_data_X_train, columns=encoder.get_feature_names_out(lCategorical_cols))
            
            # Index resetten, um Zeilen korrekt zu alignen
            encoded_X_train = encoded_X_train.reset_index(drop=True)
            X_train = X_train.drop(columns=lCategorical_cols).reset_index(drop=True)

            # Nur Spalten hinzufügen, die noch nicht existieren
            encoded_X_train = encoded_X_train.loc[:, ~encoded_X_train.columns.isin(X_train.columns)]

            # Zusammenfügen
            X_train = pd.concat([X_train, encoded_X_train], axis=1)

        if X_test is not None:
            encoded_data_X_test = encoder.transform(X_test[lCategorical_cols])
            encoded_X_test = pd.DataFrame(encoded_data_X_test, columns=encoder.get_feature_names_out(lCategorical_cols))
            
            encoded_X_test = encoded_X_test.reset_index(drop=True)
            X_test = X_test.drop(columns=lCategorical_cols).reset_index(drop=True)

            encoded_X_test = encoded_X_test.loc[:, ~encoded_X_test.columns.isin(X_test.columns)]

            X_test = pd.concat([X_test, encoded_X_test], axis=1)
        
        return X_train, X_test

    def encode_using_trigonometric_encoder(self, X_train, X_test, lCol, lMax_val):
        if X_train is not None:
            for col, max_val in zip(lCol, lMax_val):
                X_train[col + '_sin'] = np.sin(2 * np.pi * X_train[col]/max_val)
                X_train[col + '_cos'] = np.cos(2 * np.pi * X_train[col]/max_val)
                X_test[col + '_sin'] = np.sin(2 * np.pi * X_test[col]/max_val)
                X_test[col + '_cos'] = np.cos(2 * np.pi * X_test[col]/max_val)

            # keep only the encoded col in the data:
            X_train.drop(columns=lCol, inplace=True)
            X_test.drop(columns=lCol, inplace=True)

            return X_train, X_test
        

    def check_qualitative_correlation_with_chi2_and_Vcramer(self, data, column1, column2, category=None):
            """
            Calculate Cramér's V to measure the association between two categorical variables.

            Cramér's V is a statistic used to assess the strength of association between
            two nominal categorical variables. It ranges from 0 (no association) to 1 (perfect association).

            Parameters
            ----------
            column1 : str
                The name of the first categorical variable column.
            column2 : str
                The name of the second categorical variable column.
            category : str, optional
                If specified, calculates Cramér's V between column1 and a specific category of column2.

            Returns
            -------
            float
                Cramér's V value indicating the strength of association between the two variables.
            float
                The p-value from the Chi-squared test.

            Examples
            --------
            >>> cramer_v, p_value = analyzer.v_cramer('gender', 'purchase')

            Notes:
            if cramer_v <= 0.1: weak association
            if 0.2 <= cramer_v < 0.3: medium association
            if cramer_v >= 0.5: strong association
            """
            from scipy.stats import chi2_contingency
            import numpy as np

            if category is not None:
                contingency_table = pd.crosstab(data[column2], pd.get_dummies(data[column1])[category])
            else:
                contingency_table = pd.crosstab(data[column1], data[column2])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            #n = contingency_table.sum().sum()
            #min_dim = min(contingency_table.shape) - 1
            cramer_v = np.sqrt(chi2/pd.crosstab(data[column1], data[column2]).values.sum())
        
            return cramer_v, p_value
        

    def check_relationship_between_2_qualitative_variables(self, data, column1, column2):
            """
            Check the relationship between two qualitative variables using Chi-squared test.

            This method generates a contingency table and performs a Chi-squared test
            to assess the independence between the two categorical variables.

            Parameters
            ----------
            column1 : str
                The name of the first qualitative variable column.
            column2 : str
                The name of the second qualitative variable column.

            Returns
            H0: The variable X is independent of Y
            H1: X is not independent of Y
            H0 is rejected if p-value < alpha (significance level, e.g., 0.05)
            H0 is not rejected if p-value ≥ alpha   

            -------
            dict
                A dictionary containing:
                - 'contingency_table' : pd.DataFrame
                    The contingency table showing the frequency distribution.
                - 'chi2_statistic' : float
                    The Chi-squared statistic value.
                - 'p_value' : float
                    The p-value from the Chi-squared test.
                - 'degrees_of_freedom' : int
                    The degrees of freedom for the test.
                - 'expected_frequencies' : np.ndarray
                    The expected frequencies under the null hypothesis.

            Examples
            --------
            >>> result = analyzer.check_relationship_between_2_qualitative_variables('gender', 'purchase')        
            """        
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(data[column1], data[column2], normalize=True)
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            result = {
                'contingency_table': contingency_table,
                'chi2_statistic': chi2,
                'p_value': p,
                'degrees_of_freedom': dof,
                'expected_frequencies': expected
            }
            return result
    
    def check_relationship_between_quantitative_and_qualitative_variables(self, data, cat_var, num_var):
        """
        Check the relationship between a quantitative and a qualitative variable using ANOVA test.

        This method performs a one-way ANOVA test to determine if there is a statistically 
        significant influence of a qualitative variable on a quantitative variable.

        Hypotheses:
            H0: There is no significant influence of the qualitative variable on the quantitative variable.
            H1: There is a significant influence of the qualitative variable on the quantitative variable.

        Parameters:
            column1 (str): The name of the qualitative (categorical) variable.
            column2 (str): The name of the quantitative (numerical) variable.

        Returns:
            pandas.DataFrame: An ANOVA table containing the analysis of variance results,
                             including F-statistic and p-value (PR(>F)).

        Example:
            result = statsmodels.formula.api.ols('Quantitative variable ~ Qualitative variable', data=df).fit()
            statsmodels.api.stats.anova_lm(result)

        Note:
            If the p-value (PR(>F)) is less than 0.05 (5% significance level), 
            we reject the null hypothesis (H0) and conclude that there is a significant 
            influence of the qualitative variable on the quantitative variable (H1).
        """
        
        ## ANOVA test :
        import statsmodels.api 
        result = statsmodels.formula.api.ols(f'{num_var} ~ {cat_var}', data=data).fit()
        table = statsmodels.api.stats.anova_lm(result)
        return table

    def show_reports(self, y_test, y_preds, output_dict=False):
            """
            Display classification performance reports.
            Prints a classification report and a normalized confusion matrix to evaluate
            the performance of a classification model.
            Parameters
            ----------
            y_test : array-like of shape (n_samples,)
                True target values from the test set.
            y_preds : array-like of shape (n_samples,)
                Predicted target values from the model.
            Returns
            -------
            None
                Prints reports directly to the console.
            Notes
            -----
            The confusion matrix is normalized by columns (predicted class), showing
            the proportion of each predicted class that corresponds to each real class.
            """        
            return {
            "classification_report": classification_report(y_test, y_preds, output_dict=output_dict),
            "confusion_matrix": pd.crosstab(y_test, y_preds, margins=True, 
                            rownames=["Real class"], colnames=["Predicted class"])
            }
    
    def train(self,classifier, X_train, y_train):
        clf = self.get_model_instance(classifier)
        model_ = clf()
        #st.write(f"{type(classifier)} with default parameters")
        model_.fit(X_train, y_train)
        #if classifier == 'Random Forest':
            #clf = RandomForestClassifier()
        #elif classifier == 'SVC':
            #clf = SVC()
        #elif classifier == 'Logistic Regression':
            #st.write("Logistic Regression with default parameters")
            #clf = LogisticRegression()
        #clf.fit(X_train, y_train)
        return model_

    def scores(self, clf, X_test, y_test, output_dict=False):
        ##if choice == 'Accuracy':
        #   return clf.score(X_test, y_test)
        #elif choice == 'Confusion matrix':
        #   return confusion_matrix(y_test, clf.predict(X_test))
        y_pred = clf.predict(X_test)
        #st.write(f"Shape of y_pred: {y_pred.shape}")
        #st.write(f"Shape of y_test: {y_test.shape}")
        return self.show_reports(y_test, y_pred, output_dict=output_dict)
    

    def crossvalidation_resample_sfk(self, X, y, model, dResult, n_splits=5):
        """
        Performs cross-validation with multiple resampling strategies to evaluate model performance.
        This method evaluates a model using stratified k-fold cross-validation combined with three
        different resampling techniques to handle class imbalance: SMOTE, Random Oversampling, and
        Random Undersampling. For each strategy, it computes F1-scores across all folds.
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for training and testing.
        y : pd.Series
            Target variable for training and testing.
        model : estimator object
            A scikit-learn compatible model with fit() and predict() methods.
        Returns
        -------
        None
            Prints evaluation metrics (F1-scores, mean F1-score) for each resampling strategy
            to the console.
        Notes
        -----
        - Uses StratifiedKFold with 5 splits to maintain class distribution across folds.
        - Resampling is applied only to training data to prevent data leakage.
        - Three resampling strategies are evaluated:
            1. SMOTE: Synthetic Minority Over-sampling Technique
            2. Oversampling: Random oversampling of minority class
            3. Undersampling: Random undersampling of majority class
        Examples
        --------
        >>> model = LogisticRegression()
        >>> cv_resample = crossvalidation_resample(X, y, model)
        """
        
        print()
        f1_score_all = {}
        accuracy_all = {}
        for name, resample in {"Smote": SMOTE(), "Oversampling": RandomOverSampler(sampling_strategy='not majority'), "Undersampling": RandomUnderSampler(sampling_strategy='majority')}.items():

            skf = StratifiedKFold(n_splits=n_splits)

            f_score = []
            accuracy = []

            lROC_AUC = []
            lAP = []
            #lf1 = []

            print(name, end="\n")

            for train_index, test_index, in skf.split(X, y):

                X_train_, y_train_ = X.iloc[train_index], y.iloc[train_index]

                X_test_, y_test_ = X.iloc[test_index], y.iloc[test_index]

                X_train_smote, y_train_smote = resample.fit_resample(
                    X_train_, y_train_)

                model = model

                model.fit(X_train_smote, y_train_smote)

                y_pred_ = model.predict(X_test_)
                #self.show_reports(y_test_, y_pred_)

                f_score.append(f1_score(y_test_, y_pred_))
                accuracy.append(model.score(X_test_, y_test_))

                lROC_AUC.append(roc_auc_score(y_test_, y_pred_))
                lAP.append(average_precision_score(y_test_, y_pred_))
                #lf1.append(f1_score(y_pred,y_test))
            
            model_name = model.__class__.__name__
            dResult.setdefault(model_name + '_' + name, [np.mean(lROC_AUC), np.mean(lAP), np.mean(f_score)])
            #print("The F1 scores : ", end="\n")
                
            #print([round(f,2) for f in f_score], end="\n")

            #print('F1-Score average=%.5f' %
                #(np.mean(f_score)), end="\n\n")
            # add the average in the dict to be compared to catch the best method:
            f1_score_all.setdefault(resample, np.mean(f_score))
            
            #print("The ROC AUC scores : ", end="\n")
                
            #print([round(f,2) for f in lROC_AUC], end="\n")

            #print('ROC AUC average=%.5f' %
                #(np.mean(lROC_AUC)), end="\n\n")
            
            #
            #print("The average precison scores : ", end="\n")
                
            #print([round(f,2) for f in lAP], end="\n")

            #print('AP AUC average=%.5f' %
                #(np.mean(lAP)), end="\n\n")

            #print("The accuracy scores : ", end="\n")

            #print([round(f,2) for f in accuracy], end="\n")

            #print('Accuracy average=%.5f' %
                #(np.mean(accuracy)), end="\n")
            # add the average in the dict to be compared to catch the best method:
            accuracy_all.setdefault(resample, np.mean(accuracy))
            #print('########################################################################################')
            
            #return np.mean(f_score), np.std(f_score), np.min(f_score), np.max(f_score)
        
        return f1_score_all, accuracy_all    

    def calculate_probabilities(self, model, X_test):
        """
        Calculate predicted probabilities for the positive class using a trained model.
        This method takes a trained classification model and a test dataset, and returns
        the predicted probabilities for the positive class (class 1) for each instance in the test set.
        Parameters
        ----------
        model : estimator object
            A trained scikit-learn compatible classification model with a predict_proba() method.
        X_test : pd.DataFrame or np.ndarray
            The feature matrix for the test dataset.
        Returns
        -------
        np.ndarray
            array of predicted probabilities for all classes for each instance in X_test.
        np.ndarray
            an array of predicted probabilities for the positive class (class 1) for each instance in X_test.
        Examples
        --------
        >>> probabilities = calculate_probabilities(trained_model, X_test)
        """
        y_proba_all = model.predict_proba(X_test)
        y_probal = model.predict_proba(X_test)[:, 1]
        return y_proba_all, y_probal
    
    def load_state_map_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required_cols = {"state", "fraud_cases", "losses_usd"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns in map dataset: {missing}")

        df = df.copy()
        df["state"] = df["state"].astype(str).str.strip().str.upper()
        df["fraud_cases"] = pd.to_numeric(df["fraud_cases"], errors="coerce").fillna(0)
        df["losses_usd"] = pd.to_numeric(df["losses_usd"], errors="coerce").fillna(0)
        return df


    
