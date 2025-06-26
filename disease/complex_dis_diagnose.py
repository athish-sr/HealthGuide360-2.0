import joblib        #Loading the model
import pandas as pd  #Data handling
import json          #Reading a json file

disease_data={}
user_input=set()
prev=None
prev_prev=None
asked=set()
count=0
overall_count=0
yes_reduction_factor=0
no_reduction_factor=0

data=pd.read_csv("disease\\complex_diseases.csv", encoding='ISO-8859-1')

#Independent-X
X=data.drop(columns=['Disease'], axis='columns')
symptoms=list(X.columns)
#Saving the model
with open("disease\\complex_disease_data.json",'r') as f:
    old_disease_data=json.load(f)    
f.close()

disease_data={}
for k,v in old_disease_data.items():
    k_new=k.lower()
    v_new=[]
    for i in v:
        v_new.append(i.lower())
    disease_data[k_new]=v_new


#Loading the Classifier model built based on RandomForest method of classification
rf_classifier=joblib.load('disease\\complex_disease_diagnoser.pkl')

with open("disease\\diseases_info.json",'r') as file:
    disease_info=json.load(file)
file.close()

def sympt_vect(user_input, symptoms):  #Vector creator function

    #Creating symptom_dict with symptoms as key and 0 as value
    symptom_dict={symptom.lower(): 0 for symptom in symptoms}

    #Iterating through user input of symptoms
    for symptom in user_input:

        #Format input
        symptom=symptom.strip()

        #Check if user input present in symptom_dict and assigns value as 1 ,i.e. it should be a valid symptom
        if symptom in symptom_dict:
            symptom_dict[symptom]=1

            
    return [symptom_dict[symptom.lower()] for symptom in symptoms]

def prob_format(probabilities,class_labels):   #Disease probability formatter function

    #Disease probabilty dictionary
    disease_prob={}

    #Iterate with zip func
    for disease,probability in zip(class_labels, probabilities):

        #Adding only those diseases having non-zero probability
        if probability>0:
            disease_prob[disease]=probability

    #Sort the disease_prob dictionary in the increasing order of probability value
    disease_prob=dict(sorted(disease_prob.items(),key=lambda item:item[1],reverse=True))
    return disease_prob

def disease_conclude(disease,probability):  #disease conclusion provider
    return_info= f"You may have \"{disease}\" \n\n"
    return_info+=f"About: {disease_info[disease]["About"]}\n\n"
    return_info+=f"Medication: {disease_info[disease]["Medication"]}\n"
    return_info+=f"Note: Please consider consulting with your medical consultant for the medications and dosage.\n\n"
    return_info+=f"Medical Guidance: {disease_info[disease]["Medical Guidance"]}"
    return return_info

def pred(symptoms,user_input):  #Disease Predictor function
    global asked
    global prev
    global prev_prev
    global count
    global overall_count
    global yes_reduction_factor
    global no_reduction_factor
    #Vectorize user input
    user_input_vector=sympt_vect(user_input, symptoms)

    #User input dataframe
    user_input_df=pd.DataFrame([user_input_vector], columns=symptoms)

    #Probabilities Finder
    probabilities=rf_classifier.predict_proba(user_input_df)[0]

    #Getting class labels
    class_labels=rf_classifier.classes_

    #Format Probability
    disease_prob=prob_format(probabilities,class_labels)
    flag=0
    overall_count=0
    #Iterate through possible disease and symptoms
    for disease,probability in disease_prob.items():
        if overall_count>15:
            prev=0
            prev_prev=0
            count=0
            overall_count=0
            asked=set()
            return "Sorry we cant conclude..."
        #Conclude
        if probability>(0.6-yes_reduction_factor-no_reduction_factor):
            overall_count=0
            user_input=set()
            prev=None
            prev_prev=None
            asked=set()
            count=0
            overall_count=0
            yes_reduction_factor=0
            no_reduction_factor=0
            return disease_conclude(disease,probability)
        else:
            #Ask symptom for each disease
            for sympt in disease_data[disease]:
                if count>=1:
                    count=0
                    break
                #Check if exists
                if (sympt not in user_input) and (sympt not in asked):
                    asked.add(sympt)
                    overall_count+=1
                    count+=1
                    prev=sympt
                    return f"Do you have '{sympt}'"
    asked=set()
    prev=None
    prev_prev=None
    count=0
    overall_count=0
    return  "Sorry,We cant conclude..."

def main(symptoms_inp): #Main Function
    global user_input
    global prev_prev
    global prev
    for i in symptoms_inp:
        user_input.add(i)
    prev_prev=symptoms_inp
    if len(user_input)<2:
        prev=None
        prev_prev=None
        return "Insufficient data,please provide atleast one another symptom."
        
    #Predict
    return pred(symptoms,user_input)

# Train
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
    #Read
    data=pd.read_csv("C:\\Users\\valan\\OneDrive\\Documents\\archive (2)[1]\\Training.csv")

    #Independent-X Dependent-y
    X=data.drop(columns=['Disease'],axis='columns')
y=data['Disease']
`
#Splitting data for testing and training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Classifier
rf_classifier=RandomForestClassifier(n_estimators=100,random_state=42)
rf_classifier.fit(X_train,y_train)

#Predict
y_pred=rf_classifier.predict(X_test)

#Accuracy,Classification Report,Confusion
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('Classification Report of the model:')
print(classification_report(y_test,y_pred))
print('Confusion Matrix of the model:')
print(confusion_matrix(y_test,y_pred))

#Dump/Save model
joblib.dump(rf_classifier,'C:\\Users\\valan\\Downloads\\archive (1)\\rf_classifier_model.pkl')
'''
