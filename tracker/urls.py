from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'tracker'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/exercises/', views.ExerciseList.as_view(), name='api-exercises'),
    path('api/patients/', views.PatientListCreate.as_view(), name='api-patients'),
    path('patients/', views.PatientListView.as_view(), name='patient-list'),
    path('exercises/', views.hand_tracking, name='hand_tracking'),
    path('video_feed/<int:exercise_id>/', views.video_feed, name='video_feed'),
    path('interaction/', views.handle_interaction, name='handle_interaction'),
    path('get_rep_count/<int:exercise_id>/', views.get_rep_count, name='get_rep_count'),
    path('dispatch/', views.dispatch_user, name='dispatch_user'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='tracker/login.html'), name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),
    path('welcome/', views.welcome, name='welcome')
]
