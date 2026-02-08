from main import router

from ninja import NinjaAPI
from openapi_spec_validator import validate_spec

api = NinjaAPI()
api.add_router("/", router)


def test_valid_schema():
    validate_spec(api.get_openapi_schema())