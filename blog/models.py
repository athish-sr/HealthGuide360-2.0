from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='blog\\static\\images\\upload')  # Save images in the 'images/' directory
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Medicine(models.Model):
    sub_category = models.CharField(max_length=500)
    product_name = models.CharField(max_length=500)
    salt_composition = models.TextField()
    product_price = models.CharField(max_length=50,null=True)
    product_manufactured = models.CharField(max_length=500)
    medicine_desc = models.TextField()
    side_effects = models.TextField()
    drug_interactions = models.JSONField()

    def __str__(self):
        return (
            f"Sub-Category: {self.sub_category}\n"
            f"Product Name: {self.product_name}\n"
            f"Salt Composition: {self.salt_composition}\n"
            f"Product Price: {self.product_price}\n"
            f"Product Manufactured: {self.product_manufactured}\n"
            f"Medicine Description: {self.medicine_desc}\n"
            f"Side Effects: {self.side_effects}\n"
            f"Drug Interactions: {self.drug_interactions}"
        )

class Department(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Doctor(models.Model):
    name = models.CharField(max_length=100)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    # New field to handle available time slots
    time_slots = models.JSONField(default=list)  # Example format: ['09:00', '10:00']

    def __str__(self):
        return self.name

class Book(models.Model):
    name = models.CharField(max_length=100)
    phone_no = models.CharField(max_length=15)
    email=models.EmailField(default="")
    date = models.DateField()
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    time = models.TimeField()
    symptoms = models.TextField()

    def __str__(self):
        return f"{self.name} - {self.date} - {self.time}"