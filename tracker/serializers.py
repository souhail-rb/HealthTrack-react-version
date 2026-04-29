from rest_framework import serializers
from .models import Exercise, Patient

class ExerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exercise
        fields = '__all__' # Includes activity_type, duration_minutes, etc.


class PatientSerializer(serializers.ModelSerializer):
    exercises = serializers.PrimaryKeyRelatedField(
        queryset=Exercise.objects.all(), many=True, required=False
    )

    class Meta:
        model = Patient
        fields = '__all__'