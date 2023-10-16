from django.db import models

# Create your models here.
class Project(models.Model):
    projecName = models.CharField(verbose_name="Project name", max_length=30)
    labelType_choices = (
        (1, "Image"),
        (2, "Video"),
    )
    labelType = models.SmallIntegerField(verbose_name="Label Type", choices=labelType_choices, default=1)
    def __str__(self):
        return self.projecName
    