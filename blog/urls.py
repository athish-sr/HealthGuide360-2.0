from django.urls import path
from . import views

urlpatterns=[
    path("",views.index,name="index"),
    path("disease/",views.disease,name="disease"),
    path("mental/",views.mental,name="mental"),
    path("hospitals_nearby/",views.hospitals_nearby,name="hospitals_nearby"),
    path("blood_cancer/",views.blood_cancer,name="blood_cancer"),
    path("brain_cancer/",views.brain_cancer,name="brain_cancer"),
    path("kidney_cancer/",views.kidney_cancer,name="kidney_cancer"),
    path("lung_cancer/",views.lung_cancer,name="lung_cancer"),
    path("skin_cancer/",views.skin_cancer,name="skin_cancer"),
    path("result/",views.result,name="result"),
    path('heart/', views.heart_predict, name='heart_predict'),
    path('lung/', views.lung_predict, name='lung_predict'),
    path('diabetes/', views.diabetes_predict, name='diabetes_predict'),
    path('stroke/',views.stroke_predict,name='stroke_predict'),
    path('kidney_stone/', views.kidney_stone_predict, name='predict_kidney_stone'),
    path('liver_disease/',views.liver_disease_predict,name='liver_disease_predict'),
    path('prob_display/',views.prob_display,name='display_probability'),
    path('medicine/', views.medicine_list, name='medicine_list'),
    path('search/', views.search_medicine, name='search_medicine'),
    path('medicine/<int:id>/', views.medicine_detail, name='medicine_detail'),
    path('get-departments/', views.get_departments, name='get_departments'),
    path('get-doctors/', views.get_doctors, name='get_doctors'),
    path('available-times/', views.get_available_times, name='get_available_times'),
]