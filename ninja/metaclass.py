import inspect
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Type,
    TypeVar,
    Union,
    no_type_check,
    Optional,
    List,
)

import pydantic
from django.db.models import Manager, QuerySet
from django.db.models.fields.files import FieldFile
from django.template import Variable, VariableDoesNotExist
from pydantic import Field
from pydantic._internal._model_construction import ModelMetaclass
# from pydantic.functional_validators import ModelWrapValidatorHandler
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform
from django.db.models import Model as DjangoModel

from ninja.signature.utils import get_args_names, has_kwargs
# from ninja.types import DictStrAny
from ninja.errors import ConfigError
from ninja.orm.factory import factory

pydantic_version = list(map(int, pydantic.VERSION.split(".")[:2]))
assert pydantic_version >= [2, 0], "Pydantic 2.0+ required"

from devtools import debug

