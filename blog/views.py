from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.contrib import messages
from django.shortcuts import render, redirect,get_object_or_404
from django.core.mail import send_mail
from .forms import ImageUploadForm,LungForm,LiverDiseaseForm,HeartForm,StrokeForm,DiabetesForm,KidneyStoneForm,BookForm
from PIL import Image
from CancerDiagnoser import Prediction
from FuturePrediction import future_prediction
from .models import Medicine,Department,Doctor,Book
import json
from mental_health import testing
from disease import chatbot_diagnose

def index(request):
    testing.string_sug_ask=''
    testing.prev_prev=None
    testing.prev=None
    testing.asked=set()
    testing.count_tries=0
    testing.user_input=[]
    if request.method == 'POST':
        form = BookForm(request.POST)
        if form.is_valid():
            # Check availability before saving
            name=form.cleaned_data['name']
            user_email = form.cleaned_data['email']
            doctor = form.cleaned_data['doctor']
            time = form.cleaned_data['time']
            date = form.cleaned_data['date']
            
            # Check if the time slot is available
            if Book.objects.filter(doctor=doctor, date=date, time=time).exists():
                messages.error(request, 'This time slot is already booked.')
                return render(request, 'index.html', {'form': form})
            
            # Save the booking
            booking = form.save()
            
            # Send email with booking details
            subject = 'Booking Confirmation'
            message = f"Dear {name},\n\nYour booking has been confirmed.\n\nDetails:\nDoctor: {doctor}\nDate: {date}\nTime: {time}\n\nThank you for approaching us."
            from_email = 'technicalhealthguide360@gmail.com'
            recipient_list = [user_email]
            
            send_mail(subject, message, from_email, recipient_list, fail_silently=False)
            
            # Reset the form and show success message
            form = BookForm()
            messages.success(request, 'Booking Submitted and email sent.')
            return render(request, 'index.html', {'form': form})
        else:
            print("Form errors:", form.errors)
            return render(request, 'index.html', {'form': form})
    else:
        form = BookForm()
    
    return render(request, 'index.html', {'form': form})
def result(request):
    return render(request,'result.html')
def disease(request):
    testing.string_sug_ask=''
    testing.prev_prev=None
    testing.prev=None
    testing.asked=set()
    testing.count_tries=0
    testing.user_input=[]
    if request.method == "POST":
        # Extract the message from the POST request
        message = request.POST.get("message")
        
        # # Process the message and generate a response
        # # Here you can add logic to handle the message and create a response
        response_message = chatbot_diagnose.predictor(message)  # Example response
        # # Return the response as JSON
        return JsonResponse({'response': response_message})
    return render(request,'disease.html')
def mental(request):
    if request.method == "POST":
        # Extract the message from the POST request
        message = request.POST.get("message")
        
        # Process the message and generate a response
        # Here you can add logic to handle the message and create a response
        response_message = testing.predictor(message)  # Example response
        # Return the response as JSON
        return JsonResponse({'response': response_message})
    return render(request,'mental.html')
def brain_cancer(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # Optionally, open and process the uploaded image using Pillow
            img = Image.open(image_instance.image.path)

            # Do any image processing here (e.g., resizing, filtering)
            img = img.resize((300, 300))  # Resize as an example

            # Save the processed image back (overwrites the original)
            img.save(image_instance.image.path)

            result=Prediction.predict_brain_cancer(image_instance.image.path)
            if result[1] > 0.75:
                type=f'Brain Tumor Prediction: {result[0]} (Confidence: {result[1]:.2f})'
            else:
                type=f'Brain Tumor Prediction: Confidence too low '
            return render(request,'result.html',{'type':type,'details':result[2]})# Redirect to a success page
    else:
        form = ImageUploadForm()
    return render(request,'brain_cancer.html', {'form': form})
def blood_cancer(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # Optionally, open and process the uploaded image using Pillow
            img = Image.open(image_instance.image.path)

            # Do any image processing here (e.g., resizing, filtering)
            img = img.resize((300, 300))  # Resize as an example

            # Save the processed image back (overwrites the original)
            img.save(image_instance.image.path)

            result=Prediction.predict_blood_cancer(image_instance.image.path)
            if result[1] > 0.70:
                type=f'Blood Cancer Prediction: {result[0]} (Confidence: {result[1]:.2f})'
            else:
                type=f'Blood Cancer Prediction: Confidence too low'
            return render(request,'result.html',{'type':type,'details':result[2]})# Redirect to a success page
    else:
        form = ImageUploadForm()
    return render(request,'blood_cancer.html', {'form': form})
def lung_cancer(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # Optionally, open and process the uploaded image using Pillow
            img = Image.open(image_instance.image.path)

            # Do any image processing here (e.g., resizing, filtering)
            img = img.resize((300, 300))  # Resize as an example

            # Save the processed image back (overwrites the original)
            img.save(image_instance.image.path)
            result=Prediction.predict_lung_cancer(image_instance.image.path)
            if result[1] > 0.80:
                type=f'Lung Cancer Prediction: {result[0]} (Confidence: {result[1]:.2f})'
            else:
                type=f'Lung Cancer Prediction: Confidence too low'
            return render(request,'result.html',{'type':type,'details':result[2]})# Redirect to a success page
    else:
        form = ImageUploadForm()
    return render(request,'lung_cancer.html', {'form': form})
def kidney_cancer(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # Optionally, open and process the uploaded image using Pillow
            img = Image.open(image_instance.image.path)

            # Do any image processing here (e.g., resizing, filtering)
            img = img.resize((300, 300))  # Resize as an example

            # Save the processed image back (overwrites the original)
            img.save(image_instance.image.path)

            result=Prediction.predict_kidney_cancer(image_instance.image.path)
            if result[1] > 0.80:
                type=f'Kidney Cancer Prediction: {result[0]} (Confidence: {result[1]:.2f})'
            else:
                type=f'Kidney Cancer Prediction: Confidence too low'
            return render(request,'result.html',{'type':type,'details':result[2]})# Redirect to a success page
    else:
        form = ImageUploadForm()
    return render(request,'kidney_cancer.html', {'form': form})
def skin_cancer(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()

            # Optionally, open and process the uploaded image using Pillow
            img = Image.open(image_instance.image.path)

            # Do any image processing here (e.g., resizing, filtering)
            img = img.resize((300, 300))  # Resize as an example

            # Save the processed image back (overwrites the original)
            img.save(image_instance.image.path)

            result=Prediction.predict_skin_cancer(image_instance.image.path)
            if result[1] > 0.70:
                type=f'Skin Cancer Prediction: {result[0]} (Confidence: {result[1]:.2f})'
            elif result[1]<0.30:
                type=f'Skin Cancer Prediction: Benign (Confidence: {(1-result[1]):.2f})'
            else:
                type=f'Skin Cancer Prediction: Confidence too low'
            return render(request,'result.html',{'type':type,'details':result[2]})# Redirect to a success page
    else:
        form = ImageUploadForm()
    return render(request,'skin_cancer.html', {'form': form})
def hospitals_nearby(request):
    return render(request,'hospitals_nearby.html')
def prob_display(request):
    return render(request,'prob_predict.html')
def heart_predict(request):
    if request.method == 'POST':
        form = HeartForm(request.POST)
        if form.is_valid():
            # Process the data in form.cleaned_data
            # e.g, save the data to the database or perform calculations
            cleaned_data = form.cleaned_data

            # Extract individual fields from cleaned_data
            age = cleaned_data.get('age')
            sex = cleaned_data.get('sex')
            chest_pain = cleaned_data.get('chest_pain')
            bp = cleaned_data.get('bp')
            cholesterol = cleaned_data.get('cholesterol')
            fbs = cleaned_data.get('fbs')
            ekg = cleaned_data.get('ekg')
            max_hr = cleaned_data.get('max_hr')
            exercise_angina = cleaned_data.get('exercise_angina')
            st_depression = cleaned_data.get('st_depression')
            slope = cleaned_data.get('slope')
            num_vessels = cleaned_data.get('num_vessels')
            thallium = cleaned_data.get('thallium')

            input={'Age': age,                   # Older age increases risk
                    'Sex': sex,                    # Male, generally at higher risk
                    'Chest pain type': chest_pain,        # Typical angina
                    'BP': bp,                   # Elevated blood pressure
                    'Cholesterol': cholesterol,          # High cholesterol
                    'FBS over 120': fbs,           # Fasting blood sugar over 120 mg/dl
                    'EKG results': ekg,            # Abnormal EKG results
                    'Max HR': max_hr,               # Low maximum heart rate
                    'Exercise angina': exercise_angina,        # Exercise induced angina
                    'ST depression': st_depression,        # High ST depression
                    'Slope of ST': slope,            # Downsloping
                    'Number of vessels fluro': num_vessels,# More vessels showing fluoroscopy
                    'Thallium': thallium
                    }
            prediction=int(future_prediction.predict_heart_disease(input)*100)
            if prediction>50:
                guidance="Seek a cardiologist's advice for a detailed assessment. Tests such as an electrocardiogram (ECG), echocardiogram, or stress test may be recommended to evaluate heart function and determine necessary treatment."
            else:
                guidance=" Maintain a healthy diet low in saturated fats and cholesterol, Engage in regular physical activity, Avoid smoking and limit alcohol intake, Manage stress and control chronic conditions such as hypertension and diabetes."
            
            context = {
                'prediction': prediction,
                'disease_title': 'Heart Disease Probabilty',  # Customize as needed
                'description':"Heart disease encompasses various conditions that affect the heart's function, potentially leading to problems like reduced blood flow, heart attacks, or heart failure.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)
    else:
        form = HeartForm()

    return render(request, 'heart.html', {'form': form})
def lung_predict(request):
    if request.method == 'POST':
        form = LungForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from the form
            cleaned_data = form.cleaned_data

            # Prepare the input dictionary for the prediction function
            input_data = {
                'GENDER': cleaned_data.get('gender'),
                'AGE': cleaned_data.get('age'),
                'SMOKING': cleaned_data.get('smoking'),
                'YELLOW_FINGERS': cleaned_data.get('yellow_fingers'),
                'ANXIETY': cleaned_data.get('anxiety'),
                'PEER_PRESSURE': cleaned_data.get('peer_pressure'),
                'CHRONIC DISEASE': cleaned_data.get('chronic_disease'),
                'FATIGUE': cleaned_data.get('fatigue'),
                'ALLERGY': cleaned_data.get('allergy'),
                'WHEEZING': cleaned_data.get('wheezing'),
                'ALCOHOL CONSUMING': cleaned_data.get('alcohol_consuming'),
                'COUGHING': cleaned_data.get('coughing'),
                'SHORTNESS OF BREATH': cleaned_data.get('shortness_of_breath'),
                'SWALLOWING DIFFICULTY': cleaned_data.get('swallowing_difficulty'),
                'CHEST PAIN': cleaned_data.get('chest_pain'),
            }

            # Call the prediction function with the input data
            prediction = int(future_prediction.predict_lung_cancer(input_data)*100)
            if prediction>50:
                guidance="Consult a healthcare professional for a thorough evaluation. This may include pulmonary function tests, chest X-rays, or CT scans to assess lung health and determine appropriate treatment."
            else:
                guidance=" Avoid smoking and minimize exposure to pollutants, Get vaccinated against respiratory infections such as influenza and pneumonia, Maintain a healthy lifestyle with regular exercise and a balanced diet."

            # Return the result as an HttpResponse
            context = {
                'prediction': prediction,
                'disease_title':'Lung Disease Probability',  # Customize as needed
                'description':"Lung disease refers to a range of conditions that affect the lungs and respiratory system, leading to impaired breathing and overall lung function.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)

    else:
        form = LungForm()

    return render(request, 'lung.html', {'form': form})
def diabetes_predict(request):
    if request.method == 'POST':
        form = DiabetesForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from the form
            cleaned_data = form.cleaned_data

            # Prepare the input dictionary for the prediction function
            user_input = {
                'Pregnancies': cleaned_data.get('pregnancies'),
                'Glucose': cleaned_data.get('glucose'),
                'BloodPressure': cleaned_data.get('blood_pressure'),
                'SkinThickness': cleaned_data.get('skin_thickness'),
                'Insulin': cleaned_data.get('insulin'),
                'BMI': cleaned_data.get('bmi'),
                'DiabetesPedigreeFunction': cleaned_data.get('diabetes_pedigree_function'),
                'Age': cleaned_data.get('age'),
            }

            # Call the prediction function with the prepared input data
            prediction = int(future_prediction.predict_diabetes(user_input)*100)

            # For demonstration, print the prediction result
            print(f"Predicted Diabetes Score: {prediction}")
            if prediction>50:
                guidance="Consult an endocrinologist for evaluation. Blood glucose tests such as fasting blood sugar and HbA1c will help confirm the diagnosis and guide treatment options."
            else:
                guidance=" Follow a balanced diet with low refined sugars and high fiber, Engage in regular physical activity, Maintain a healthy weight, Monitor blood glucose levels and manage stress."
            # Return the prediction result as an HttpResponse
            context = {
                'prediction': prediction,
                'disease_title': 'Diabetes Probability',  # Customize as needed
                'description':"Diabetes is a chronic condition characterized by high blood sugar levels due to the body's inability to produce or use insulin effectively.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)

    else:
        form = DiabetesForm()

    return render(request, 'diabetes.html', {'form': form})
def stroke_predict(request):
    if request.method == 'POST':
        form = StrokeForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from the form
            cleaned_data = form.cleaned_data

            # Prepare the input dictionary for the prediction function
            user_input = {
                'gender': cleaned_data.get('gender'),
                'age': cleaned_data.get('age'),
                'hypertension': cleaned_data.get('hypertension'),
                'heart_disease': cleaned_data.get('heart_disease'),
                'ever_married': cleaned_data.get('ever_married'),
                'work_type': cleaned_data.get('work_type'),
                'Residence_type': cleaned_data.get('Residence_type'),
                'avg_glucose_level':cleaned_data.get('avg_glucose_level'),
                'bmi': cleaned_data.get('bmi'),
                'smoking_status':cleaned_data.get('smoking_status'),
            
            }
            print(user_input)
            # Call the prediction function with the prepared input data
            prediction = int(future_prediction.predict_stroke(user_input)*100)
            # Return the prediction result as an HttpResponse
            if prediction>50:
                guidance="Seek immediate medical attention for a comprehensive evaluation. This may involve brain imaging such as CT scans or MRI to assess your risk and guide preventive strategies."
            else:
                guidance="Control high blood pressure and manage diabetes, Avoid smoking and limit alcohol consumption, Maintain a healthy diet and manage cholesterol levels, Engage in regular physical activity."

            context = {
                'prediction': prediction,
                'disease_title': 'Stroke Probability',  # Customize as needed
                'description':"A stroke occurs when blood flow to a part of the brain is interrupted, which can cause brain damage and impair function.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)

    else:
        form = StrokeForm()

    return render(request, 'stroke.html', {'form': form})
def kidney_stone_predict(request):
    if request.method == 'POST':
        form = KidneyStoneForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            # Prepare user input
            user_input = {
                'gravity': cleaned_data.get('gravity'),
                'ph': cleaned_data.get('ph'),
                'osmo': cleaned_data.get('osmo'),
                'cond': cleaned_data.get('cond'),
                'urea': cleaned_data.get('urea'),
                'calc': cleaned_data.get('calc'),
            }
            # Make prediction
            prediction = int(future_prediction.predict_kidney_stone(user_input)*100)
            if prediction>50:
                guidance="Consult a urologist for evaluation. Imaging tests such as ultrasound or CT scans may be necessary to determine the size and location of any stones and plan appropriate management."
            else:
                guidance=" Stay well-hydrated by drinking plenty of water, Reduce intake of salt and oxalate-rich foods, Maintain a balanced diet with moderate calcium intake, Engage in regular exercise."

            context = {
                'prediction': prediction,
                'disease_title': 'Kidney Stone Probability',  # Customize as needed
                'description':"Kidney stones are hard deposits that form in the kidneys and can cause pain and urinary issues when they move or become lodged in the urinary tract.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)
    else:
        form = KidneyStoneForm()

    return render(request, 'kidney.html', {'form': form})
def clean_user_input(user_input):
    return {
        'Age': user_input.get('age') if user_input.get('age') is not None else 0,
        'Gender': user_input.get('gender') if user_input.get('gender') is not None else 0,
        'BMI': user_input.get('bmi') if user_input.get('bmi') is not None else 0,
        'AlcoholConsumption': user_input.get('alcohol_consumption') if user_input.get('alcohol_consumption') is not None else 0,
        'Smoking': user_input.get('smoking') if user_input.get('smoking') is not None else 0,
        'GeneticRisk': user_input.get('genetic_risk') if user_input.get('genetic_risk') is not None else 0,
        'PhysicalActivity': user_input.get('physical_activity') if user_input.get('physical_activity') is not None else 0,
        'Diabetes': user_input.get('diabetes') if user_input.get('diabetes') is not None else 0,
        'Hypertension': user_input.get('hypertension') if user_input.get('hypertension') is not None else 0,
        'LiverFunctionTest': user_input.get('liver_function_test') if user_input.get('liver_function_test') is not None else 0
    }
def liver_disease_predict(request):
    if request.method == 'POST':
        form = LiverDiseaseForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data

            user_input = {
                'Age': cleaned_data.get('age'),
                'Gender': cleaned_data.get('gender'),
                'BMI': cleaned_data.get('bmi'),
                'AlcoholConsumption': cleaned_data.get('alcohol_consumption'),
                'Smoking': cleaned_data.get('smoking'),
                'GeneticRisk': cleaned_data.get('genetic_risk'),
                'PhysicalActivity': cleaned_data.get('physical_activity'),
                'Diabetes': cleaned_data.get('diabetes'),
                'Hypertension': cleaned_data.get('hypertension'),
                'LiverFunctionTest': cleaned_data.get('liver_function_test'),
            }
            prediction = future_prediction.predict_liver_disease(user_input)*100
            if prediction>50:
                guidance="Consult a hepatologist for a thorough evaluation. This may include liver function tests, imaging studies, and possibly a liver biopsy to assess liver health and determine appropriate treatment."
            else:
                guidance=" Avoid excessive alcohol consumption, Get vaccinated for hepatitis A and B, Maintain a healthy weight and diet, Avoid exposure to harmful chemicals and practice safe sex."

            context = {
                'prediction': prediction,
                'disease_title': 'Liver Disease',
                'description':"Liver disease refers to conditions that impact liver function, which can affect overall health.",
                'guidance': guidance
            }
            return render(request,'prob_predict.html', context)
        else:
            # Handle the form errors
            return HttpResponse(f'Form errors: {form.errors}')
    else:
        form = LiverDiseaseForm()
    return render(request, 'liver.html', {'form': form})
def medicine_list(request):
    medicines = Medicine.objects.all()
    return render(request, 'medicine.html', {'medicines': medicines})
def search_medicine(request):
    query = request.GET.get('query', '')
    medicines = Medicine.objects.filter(product_name__icontains=query)
    results = list(medicines.values())
    return JsonResponse({'results': results})
def medicine_detail(request, id):
    medicine = get_object_or_404(Medicine, pk=id)
    try:
        drug_interactions = json.loads(medicine.drug_interactions) 
        drug=drug_interactions['drug']
        brand=drug_interactions['brand']
        effect=drug_interactions['effect'] # Convert JSON string to Python dictionary
        combined=zip(drug,brand,effect)
        
    except json.JSONDecodeError:
        combined = {}
    return render(request,'medicine_details.html', {'medicine': medicine,'combined':combined})
def get_departments(request):
    departments = Department.objects.all()
    data = [{'id': dept.id, 'name': dept.name} for dept in departments]
    return JsonResponse({'departments': data})

# Doctor fetcher
def get_doctors(request):
    department_id = request.GET.get('department_id')
    doctors = Doctor.objects.filter(department_id=department_id)
    data = [{'id': doc.id, 'name': doc.name, 'time_slots': doc.time_slots} for doc in doctors]
    return JsonResponse({'doctors': data})

# Time fetcher
def get_available_times(request):
    doctor_id = request.GET.get('doctor_id')
    date = request.GET.get('date')
    
    # Fetch the doctor
    doctor = Doctor.objects.get(id=doctor_id)
    
    # Get the booked times for the given doctor and date
    booked_times = Book.objects.filter(doctor=doctor, date=date).values_list('time', flat=True)
    
    # Convert booked_times (datetime.time) to string format ('%H:%M')
    booked_times_str = [time.strftime('%H:%M') for time in booked_times]
    
    # Assuming doctor.time_slots is a list of time slots as strings in 'HH:MM' format
    available_times = [slot for slot in doctor.time_slots if slot not in booked_times_str]
    print(available_times)
    return JsonResponse({'available_times': available_times})
