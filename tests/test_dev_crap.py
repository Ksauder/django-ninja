from devtools import debug

from pydantic import BaseModel

from ninja import Schema


def test_dev_schema():
    class NewBase(Schema):
        customfield: str = ""

        class Meta:
            fields = "__all__"

    class MySchema(NewBase):
        anotherfield: str = "default"

    nb = MySchema(customfield="1")

