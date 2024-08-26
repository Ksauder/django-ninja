from devtools import debug
from typing import Optional

from pydantic import BaseModel

from ninja import Schema, Field
from django.db import models


class AnotherParentModel(models.Model):
        parent_field = models.CharField()

        class Meta:
            app_label = "tests"


class DocumentModel(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.TextField()
    stuff = models.CharField()
    sensitive = models.CharField()

    class Meta:
        app_label = "tests"


def test_dev_schema():
    class SupportGetSchema(Schema):
        class Meta:
            model = AnotherParentModel
            fields = "__all__"

    # if ModelSchemaMetaclass and create_schema mirror each other properly, these could be create_schema()s
    class _DocumentSchema(Schema):
        # TODO: add some custom pydantic serializers/validators and ensure they inherit properly
        # ^ like fixing the null return issue
        custom_field: Optional[SupportGetSchema] = None

        class Meta:
            model = DocumentModel
            exclude = ["sensitive"]


    class DocumentGetSchema(_DocumentSchema):
        class Meta:
            fields = "__all__"


    class DocumentPostSchema(_DocumentSchema):
        class Meta:
            exclude = ["sensitive", "id"]


    class DocumentPatchSchema(_DocumentSchema):
        class Meta:
            fields_optional = "__all__"

    nb = DocumentGetSchema(custom_field=SupportGetSchema(parent_field="y"), name="name", stuff="morestuff")
    debug(nb)
