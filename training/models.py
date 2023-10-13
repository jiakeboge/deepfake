from django.db import models

# Create your models here.

class Detection_model(models.Model):
    modelName = models.CharField(verbose_name="Model name", max_length=50)

    def __str__(self):
        return self.modelName
    
    
