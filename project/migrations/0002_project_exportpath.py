# Generated by Django 3.2.6 on 2022-09-30 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('project', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='exportPath',
            field=models.FileField(default='', max_length=128, upload_to=''),
        ),
    ]