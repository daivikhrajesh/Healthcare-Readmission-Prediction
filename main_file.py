import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_dataset(path):

    try :
        dataframe=pd.read_csv(path)
        return dataframe
    except Exception as e:
        print(e)
        
class Preprocessing:

    def datahead(df):

        return df.head()
    
    def describe(df):
        
        return df.describe(include = 'all').T
    
    def missing(df):
        for i in df.columns:
            print(i, df[df[i] == '?'].shape[0])
            
    def drop(df):
        df = df[~((df['diag_1'] == "?") | (df['diag_2'] == "?") | (df['diag_3'] == "?"))]
        print("Dropped Successfully")
        
class Analysis:
    
    def patientinfo(df):
        try:
            print("Same patient to encounter_id : ", len(df['patient_nbr'].unique()), len(df['encounter_id'].unique()))
            print("Patient to encounter_id ratio : ", len(df['patient_nbr'].unique()) / len(df['encounter_id'].unique()))
            print("Readmitted ")
            print(df['readmitted'].value_counts())
        except KeyError as e:
            print("Error'", str(e), "'. Please provide a valid dataframe.")
        except Exception as e:
            print("An error occurred: ", str(e))


    def readmitted1(df):
        try:
            sns.countplot(x="readmitted", data=df)
            plt.show()
        except KeyError as e:
            print(f"Error: {e}.'readmitted' column not found in dataframe.")



    def convert_readmitted(df):
        try:
            df['readmitted_2'] = df['readmitted'].replace(['<30', '>30'], 'Yes')
            sns.countplot(x="readmitted_2", data=df)
            plt.show()
        except KeyError as e:
            print(f"Error: {e}. column not found in the dataframe.")
        except Exception as e:
            print(f"Error: {e}. An unexpected error occurred.")



    def race_distribution(df):
        try:
            df.loc[df['race'] == '?', 'race'] = 'Other'
            race_counts = df['race'].value_counts()
            race1 = sns.barplot(x=race_counts.index, y=race_counts)
            plt.xlabel('Race')
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.show()
        except KeyError as e:
            print(f"KeyError: {str(e)}. Please check if the DataFrame has a 'race' column.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


    def gender_readmission(df):
            df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace = True)
            gender_readmit = sns.countplot(x="gender", hue="readmitted_2", data=df)
            plt.xlabel('Gender')
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.show()


    def age_distribution(df):
            age1 = sns.countplot(x="age", hue="readmitted_2", data=df)
            plt.xlabel('Age')
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.show()


    def time_in_hospital(df):
        try:
            time_hospital = sns.countplot(x='time_in_hospital', hue='readmitted_2', data=df)
            plt.xlabel('Time In Hospital')
            plt.xticks(rotation=90)
            plt.ylabel('Readmitted Count')
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")



    def count_unique_diag(df):
            print(f"Number of unique values in diag_1: {len(df['diag_1'].unique())}")
            print(f"Number of unique values in diag_2: {len(df['diag_2'].unique())}")
            print(f"Number of unique values in diag_3: {len(df['diag_3'].unique())}")



    def other_cols(df):
            for i in df.iloc[:, 22:44].columns:
                med1 = sns.countplot(x=i, data= df)
                plt.ylabel('Count')
                plt.show()


    def plot_histograms(df):
            for j in df.iloc[:, 21:44].columns:
                z = sns.FacetGrid(df, col=j)
                z.map(sns.histplot, "readmitted_2")
                plt.show()

    def drop_columns(df):
        try:
            columns_to_drop = ['acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide', 
                               'examide', 'citoglipton', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                               'metformin-rosiglitazone', 'metformin-pioglitazone', 'weight', 'payer_code']
            df.drop(columns=columns_to_drop, inplace=True)
            print("Columns dropped successfully!")
        except KeyError:
            print("Error: One or more columns could not be dropped. Please check the column names.")

        
        
class Conversion:
   
    def label_enc(self, df):
        try:
            df1 = df.copy()
            LE = LabelEncoder()
            categorical_features =['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                           'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 
                           'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 
                      'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'change', 'diabetesMed'] 

            for i in categorical_features:
                 df1[i] = LE.fit_transform(df1[i])
            return df1
        except:
            print("Error occurred while encoding categorical features.")

    def label_enc2(self, df):
        try:
            df1 = self.label_enc(df)
            LE = LabelEncoder()
            lbl = LE.fit(df1['readmitted_2'])
            df1['readmitted_2_lbl'] = lbl.transform(df1['readmitted_2'])
            df1 = df1.drop(columns=['encounter_id', 'patient_nbr', 'readmitted', 'readmitted_2'])
            print("Successful")
            return df1
        except:
            print("Error occurred while encoding 'readmitted_2' column and dropping unnecessary columns.")





class RFC:
    
    
    def RTclassifier(df):
        try:
            X = df.drop(columns=['readmitted_2_lbl'])
            Y = df['readmitted_2_lbl']
            scaled_X = preprocessing.StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.30)
            rf = RandomForestClassifier(n_estimators=450, max_depth=9)
            rf.fit(X_train, y_train)
            rf_prediction = rf.predict(X_test)
            print(classification_report(y_test, rf_prediction, target_names=['Not Readmitted', 'Readmitted']))
            mse = mean_squared_error(y_test, rf_prediction)
            mae = mean_absolute_error(y_test, rf_prediction)


            cm = confusion_matrix(y_test, rf_prediction)

            fig, ax = plt.subplots(figsize=(15, 7))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=['Not Readmitted', 'Readmitted'],
                   yticklabels=['Not Readmitted', 'Readmitted'],
                   title='Confusion Matrix of Random Forest Model \n',
                   ylabel='True label',
                   xlabel='Predicted label')

            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "red")

            fig.tight_layout()
            plt.show()

            print('MSE:', mse)
            print('MAE:', mae)
        except Exception as e:
            print(f"An error occurred: {e}")



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        