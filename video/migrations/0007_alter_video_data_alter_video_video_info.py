# Generated by Django 4.1.2 on 2022-10-31 04:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('video', '0006_auto_20220920_0344'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='data',
            field=models.JSONField(null=True),
        ),
        migrations.AlterField(
            model_name='video',
            name='video_info',
            field=models.JSONField(null=True),
        ),
    ]
