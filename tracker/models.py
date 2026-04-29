from django.db import models

# Create your models here.

class Exercise(models.Model):
    ACTIVITY_CHOICES = [
        ('OPEN_CLOSE', 'Open/Close Hand'),
        ('PINCH', 'Thumb-Index Pinch'),
        ('SPREAD', 'Finger spread'),
        ('CURL', 'Finger curl'),
       
    ]
    #
    activity_type = models.CharField(max_length=50, choices=ACTIVITY_CHOICES)
    start_time = models.DateTimeField(auto_now_add=True)
    duration_minutes = models.IntegerField(default=0)
    reps_count = models.IntegerField(default=0)
    notes = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Workout - {self.start_time.strftime('%Y-%m-%d %H:%M')}"



class Patient(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    height = models.DecimalField(max_digits=5, decimal_places=2)
    weight = models.DecimalField(max_digits=5, decimal_places=2)
    exercises = models.ManyToManyField(Exercise)

    def __str__(self):
        return self.name
