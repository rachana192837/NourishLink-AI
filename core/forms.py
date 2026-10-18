from datetime import date

from django import forms
from django.core.exceptions import ValidationError
from django.forms import formset_factory

from .models import FoodBatch, Product, Request


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'unit', 'category', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 2}),
        }


class BatchForm(forms.ModelForm):
    class Meta:
        model = FoodBatch
        fields = ['product', 'donor', 'batch_code', 'quantity_received',
                  'received_date', 'distribution_cutoff', 'notes']
        widgets = {
            'received_date': forms.DateInput(attrs={'type': 'date'}),
            'distribution_cutoff': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 2}),
        }

    def clean(self):
        cleaned = super().clean()
        received = cleaned.get('received_date')
        cutoff = cleaned.get('distribution_cutoff')
        if received and cutoff and cutoff <= received:
            raise ValidationError(
                'Distribution cutoff must be after the received date.')
        return cleaned


class RequestForm(forms.ModelForm):
    class Meta:
        model = Request
        fields = ['required_by_date', 'notes']
        widgets = {
            'required_by_date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

    def clean_required_by_date(self):
        d = self.cleaned_data['required_by_date']
        if d < date.today():
            raise ValidationError('Required-by date cannot be in the past.')
        return d


class RequestItemForm(forms.Form):
    product = forms.ModelChoiceField(
        queryset=Product.objects.all(), empty_label='Select product')
    quantity_requested = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'min': 1, 'placeholder': 'Qty'}))


class BaseRequestItemFormSet(forms.BaseFormSet):
    def clean(self):
        if any(self.errors):
            return
        products = []
        for form in self.forms:
            if not form.cleaned_data:
                continue
            product = form.cleaned_data.get('product')
            if product and product in products:
                raise ValidationError(
                    'Duplicate products in one request are not allowed.')
            if product:
                products.append(product)


RequestItemFormSet = formset_factory(
    RequestItemForm, formset=BaseRequestItemFormSet,
    extra=3, min_num=1, validate_min=True)
