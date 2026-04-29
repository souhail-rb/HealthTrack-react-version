from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.generic import ListView
import mediapipe as mp
import cv2
from django.utils import timezone
import json
from .models import Exercise, Patient
from .camera import VideoCamera


from django.shortcuts import redirect
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required


from rest_framework import generics
from .models import Exercise
from .serializers import ExerciseSerializer, PatientSerializer
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt

class ExerciseList(generics.ListCreateAPIView):
    queryset = Exercise.objects.all()
    serializer_class = ExerciseSerializer
    permission_classes = [AllowAny]


class PatientListCreate(generics.ListCreateAPIView):
    queryset = Patient.objects.all().order_by('name')
    serializer_class = PatientSerializer
    permission_classes = [AllowAny]

# Create your views here.
def welcome(request):
    return render(request, 'tracker/welcome.html')

def home(request):
    # Query for exercises if needed, using the correct model name
    exercises = Exercise.objects.all().order_by('-start_time')
    context = {
        'mp_version': mp.__version__,
        'cv2_version': cv2.__version__,
        'exercises': exercises,
    }
    
    return render(request, 'tracker/home.html', context)

class PatientListView(ListView):
    model = Patient
    template_name = 'tracker/patient_list.html'
    context_object_name = 'patients'

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request, exercise_id):
    return StreamingHttpResponse(gen(VideoCamera(exercise_id=exercise_id)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def hand_tracking(request):
    return render(request, 'tracker/hand_tracking.html')

def get_rep_count(request, exercise_id):
    try:
        exercise = Exercise.objects.get(pk=exercise_id)
        return JsonResponse({'status': 'success', 'rep_count': exercise.reps_count})
    except Exercise.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Exercise not found'}, status=404)

@csrf_exempt
def handle_interaction(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        action = data.get('action')

        if action == 'start':
            activity_type = data.get('activity_type', 'OPEN_CLOSE')
            # Create a new exercise session
            exercise = Exercise.objects.create(activity_type=activity_type)
            print(f"Started exercise {exercise.id}")
            return JsonResponse({'status': 'success', 'exercise_id': exercise.id})

        elif action == 'stop':
            exercise_id = data.get('exercise_id')
            if not exercise_id:
                return JsonResponse({'status': 'error', 'message': 'exercise_id not provided'}, status=400)
            try:
                exercise = Exercise.objects.get(pk=exercise_id)
                # Calculate duration
                duration = timezone.now() - exercise.start_time
                exercise.duration_minutes = int(duration.total_seconds() / 60)
                exercise.save()
                print(f"Stopped exercise {exercise.id}. Reps: {exercise.reps_count}, Duration: {exercise.duration_minutes} mins")
                return JsonResponse({'status': 'success', 'message': f'Exercise {exercise.id} stopped.'})
            except Exercise.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': 'Exercise not found'}, status=404)

    return JsonResponse({'status': 'error'}, status=400)



@login_required
def dispatch_user(request):
    """Redirects users to the correct dashboard based on their role."""
    if request.user.is_staff:
        # Admins go to the Patient List
        return redirect('tracker:patient-list')
    else:
        # Patients go to the Rehabilitation/Home page
        return redirect('tracker:home')


def logout_view(request):
    logout(request)
    return redirect('tracker:login')
    

