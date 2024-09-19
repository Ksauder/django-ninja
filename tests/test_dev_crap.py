from typing import Optional

from django.db import models

from ninja import NinjaAPI, Schema
from ninja.schema import create_schema
from ninja.testing import TestClient


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

        # class Config:  # Meta does only models.Model things? Config does pydantic things?
        # stuff

    class DocumentGetSchema(_DocumentSchema):
        class Meta:
            fields = "__all__"

    class DocumentPostSchema(_DocumentSchema):
        class Meta:
            exclude = ["sensitive", "id"]

    class DocumentPatchSchema(_DocumentSchema):
        class Meta:
            fields_optional = "__all__"

    api = NinjaAPI()
    client = TestClient(api)

    @api.get("/test", response=DocumentGetSchema)
    def get_doc(request):
        return DocumentGetSchema(
            custom_field=SupportGetSchema(parent_field="y"),
            name="name",
            stuff="morestuff",
        )

    res = client.get("/test")
    assert res.json() == {
        "custom_field": {
            "id": None,
            "parent_field": "y",
        },
        "id": None,
        "name": "name",
        "stuff": "morestuff",
    }

    DynSchema = create_schema(
        DocumentModel, exclude=["sensitive"], primary_key_optional=False
    )
    nb = DocumentGetSchema(
        custom_field=SupportGetSchema(parent_field="y"), name="name", stuff="morestuff"
    )

    assert nb.dict() == {
        "custom_field": {
            "id": None,
            "parent_field": "y",
        },
        "id": None,
        "name": "name",
        "stuff": "morestuff",
    }

    ndyn = DynSchema(
        id=1,
        custom_field=SupportGetSchema(parent_field="y"),
        name="name",
        stuff="morestuff",
    )

    assert ndyn.dict() == {
        "id": 1,
        "name": "name",
        "stuff": "morestuff",
    }

    assert ndyn.json_schema() == {
        "properties": {
            "id": {
                "title": "Id",
                "type": "integer",
            },
            "name": {
                "title": "Name",
                "type": "string",
            },
            "stuff": {
                "title": "Stuff",
                "type": "string",
            },
        },
        "required": [
            "id",
            "name",
            "stuff",
        ],
        "title": "DocumentModel",
        "type": "object",
    }
    # nb2 = DocumentPostSchema(custom_field=SupportGetSchema(parent_field="y"), name="name", stuff="morestuff")
