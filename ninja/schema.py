"""
Since "Model" word would be very confusing when used in django context, this
module basically makes an alias for it named "Schema" and adds extra whistles to
be able to work with django querysets and managers.

The schema is a bit smarter than a standard pydantic Model because it can handle
dotted attributes and resolver methods. For example::


    class UserSchema(User):
        name: str
        initials: str
        boss: str = Field(None, alias="boss.first_name")

        @staticmethod
        def resolve_name(obj):
            return f"{obj.first_name} {obj.last_name}"

"""
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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union, cast
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union, no_type_check
from decimal import Decimal
import datetime
from uuid import UUID
import itertools

import pydantic
from django.db.models import Manager, QuerySet, ManyToManyRel, ManyToOneRel, Model, ManyToManyField
from django.db.models import Field as DjangoField
from django.db.models.fields.files import FieldFile
from django.template import Variable, VariableDoesNotExist
from pydantic import Field, IPvAnyAddress, BaseModel, ValidationInfo, model_validator, validator
from pydantic import create_model as create_pydantic_model
from pydantic._internal._model_construction import ModelMetaclass
# from pydantic.functional_validators import ModelWrapValidatorHandler
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic.dataclasses import dataclass
from pydantic.fields import FieldInfo
from pydantic.functional_validators import ModelWrapValidatorHandler
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import dataclass_transform
from django.db.models import Model as DjangoModel

from ninja.signature.utils import get_args_names, has_kwargs
# from ninja.types import DictStrAny
from ninja.errors import ConfigError
from ninja.types import DictStrAny

import inspect
from devtools import debug

pydantic_version = list(map(int, pydantic.VERSION.split(".")[:2]))
assert pydantic_version >= [2, 0], "Pydantic 2.0+ required"

__all__ = ["BaseModel", "Field", "validator", "DjangoGetter", "Schema"]
# __all__ = ["create_m2m_link_type", "get_schema_field", "get_related_field_schema"]

S = TypeVar("S", bound="Schema")

@dataclass
class MetaConf:
    model: Any
    fields: Optional[List[str]] = None
    exclude: Union[List[str], str, None] = None
    fields_optional: Union[List[str], str, None] = None
    depth: int = 0

    @staticmethod
    def from_schema_class(name: str, namespace: dict) -> "MetaConf":
        if "Meta" in namespace:
            meta = namespace["Meta"]
            model = meta.model
            fields = getattr(meta, "fields", None)
            exclude = getattr(meta, "exclude", None)
            optional_fields = getattr(meta, "fields_optional", None)
            depth = getattr(meta, "depth", None)

        elif "Config" in namespace:
            config = namespace["Config"]
            model = config.model
            fields = getattr(config, "model_fields", None)
            exclude = getattr(config, "model_exclude", None)
            optional_fields = getattr(config, "model_fields_optional", None)
            depth = getattr(config, "depth", None)

            warnings.warn(
                "The use of `Config` class is deprecated for ModelSchema, use 'Meta' instead",
                DeprecationWarning,
                stacklevel=2,
            )

        else:
            raise ConfigError(
                f"ModelSchema class '{name}' requires a 'Meta' (or a 'Config') subclass"
            )

        assert issubclass(model, DjangoModel)

        if not fields and not exclude:
            raise ConfigError(
                "Creating a ModelSchema without either the 'fields' attribute"
                " or the 'exclude' attribute is prohibited"
            )

        if fields == "__all__":
            fields = None
            # ^ when None is passed to create_schema - all fields are selected

        return MetaConf(
            model=model,
            fields=fields,
            exclude=exclude,
            fields_optional=optional_fields,
            depth=depth
        )

class DjangoGetter:
    __slots__ = ("_obj", "_schema_cls", "_context", "__dict__")

    def __init__(self, obj: Any, schema_cls: Type[S], context: Any = None):
        self._obj = obj
        self._schema_cls = schema_cls
        self._context = context

    def __getattr__(self, key: str) -> Any:
        # if key.startswith("__pydantic"):
        #     return getattr(self._obj, key)

        resolver = self._schema_cls._ninja_resolvers.get(key)
        if resolver:
            value = resolver(getter=self)
        else:
            if isinstance(self._obj, dict):
                if key not in self._obj:
                    raise AttributeError(key)
                value = self._obj[key]
            else:
                try:
                    value = getattr(self._obj, key)
                except AttributeError:
                    try:
                        # value = attrgetter(key)(self._obj)
                        value = Variable(key).resolve(self._obj)
                        # TODO: Variable(key) __init__ is actually slower than
                        #       Variable.resolve - so it better be cached
                    except VariableDoesNotExist as e:
                        raise AttributeError(key) from e
        return self._convert_result(value)

    # def get(self, key: Any, default: Any = None) -> Any:
    #     try:
    #         return self[key]
    #     except KeyError:
    #         return default

    def _convert_result(self, result: Any) -> Any:
        if isinstance(result, Manager):
            return list(result.all())

        elif isinstance(result, getattr(QuerySet, "__origin__", QuerySet)):
            return list(result)

        if callable(result):
            return result()

        elif isinstance(result, FieldFile):
            if not result:
                return None
            return result.url

        return result

    def __repr__(self) -> str:
        return f"<DjangoGetter: {repr(self._obj)}>"


class Resolver:
    __slots__ = ("_func", "_static", "_takes_context")
    _static: bool
    _func: Any
    _takes_context: bool

    def __init__(self, func: Union[Callable, staticmethod]):
        if isinstance(func, staticmethod):
            self._static = True
            self._func = func.__func__
        else:
            self._static = False
            self._func = func

        arg_names = get_args_names(self._func)
        self._takes_context = has_kwargs(self._func) or "context" in arg_names

    def __call__(self, getter: DjangoGetter) -> Any:
        kwargs = {}
        if self._takes_context:
            kwargs["context"] = getter._context

        if self._static:
            return self._func(getter._obj, **kwargs)
        raise NotImplementedError(
            "Non static resolves are not supported yet"
        )  # pragma: no cover
        # return self._func(self._fake_instance(getter), getter._obj)

    # def _fake_instance(self, getter: DjangoGetter) -> "Schema":
    #     """
    #     Generate a partial schema instance that can be used as the ``self``
    #     attribute of resolver functions.
    #     """

    #     class PartialSchema(Schema):
    #         def __getattr__(self, key: str) -> Any:
    #             value = getattr(getter, key)
    #             field = getter._schema_cls.model_fields[key]
    #             value = field.validate(value, values={}, loc=key, cls=None)[0]
    #             return value

    #     return PartialSchema()


@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class ResolverMetaclass(ModelMetaclass):
    _ninja_resolvers: Dict[str, Resolver]

    @no_type_check
    def __new__(cls, name, bases, namespace, **kwargs):
        resolvers = {}

        for base in reversed(bases):
            base_resolvers = getattr(base, "_ninja_resolvers", None)
            if base_resolvers:
                resolvers.update(base_resolvers)
        for attr, resolve_func in namespace.items():
            if not attr.startswith("resolve_"):
                continue
            if (
                not callable(resolve_func)
                # A staticmethod isn't directly callable in Python <=3.9.
                and not isinstance(resolve_func, staticmethod)
            ):
                continue  # pragma: no cover
            resolvers[attr[8:]] = Resolver(resolve_func)

        result = super().__new__(cls, name, bases, namespace, **kwargs)
        result._ninja_resolvers = resolvers
        return result


class ModelSchemaMetaclass(ResolverMetaclass):
    @no_type_check
    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict,
        **kwargs,
    ):
        debug(name)
        if getattr(namespace, "Meta", None):
            debug(namespace.Meta)
        if name == "MySchema":
            for base in bases:
                inspect.getmembers(base)
        # if Meta:
        #   use Meta
        #   overwrite __ninja_meta__ if exists
        # elif __ninja_meta__:
        #   
        cls = super().__new__(
            mcs,
            name,
            bases,
            namespace,
            **kwargs,
        )
        if name == "NewBase":
            debug(inspect.getmembers(cls))
        # for base in reversed(bases):
        #     if (
        #         _is_modelschema_class_defined
        #         and issubclass(base, ModelSchema)
        #         and base == ModelSchema
        #     ):
        #         meta_conf = MetaConf.from_schema_class(name, namespace)

        #         custom_fields = []
        #         annotations = namespace.get("__annotations__", {})
        #         for attr_name, type in annotations.items():
        #             if attr_name.startswith("_"):
        #                 continue
        #             default = namespace.get(attr_name, ...)
        #             custom_fields.append((attr_name, type, default))

        #         # # cls.__doc__ = namespace.get("__doc__", config.model.__doc__)
        #         # cls.__fields__ = {}  # forcing pydantic recreate
        #         # # assert False, "!! cls.model_fields"

        #         # print(config.model, name, fields, exclude, "!!")

        #         model_schema = create_schema(
        #             meta_conf.model,
        #             name=name,
        #             fields=meta_conf.fields,
        #             exclude=meta_conf.exclude,
        #             optional_fields=meta_conf.fields_optional,
        #             custom_fields=custom_fields,
        #             base_class=cls,
        #         )
        #         model_schema.__doc__ = cls.__doc__
        #         return model_schema

        return cls


class NinjaGenerateJsonSchema(GenerateJsonSchema):
    def default_schema(self, schema: Any) -> JsonSchemaValue:
        # Pydantic default actually renders null's and default_factory's
        # which really breaks swagger and django model callable defaults
        # so here we completely override behavior
        json_schema = self.generate_inner(schema["schema"])

        default = None
        if "default" in schema and schema["default"] is not None:
            default = self.encode_default(schema["default"])

        if "$ref" in json_schema:
            # Since reference schemas do not support child keys, we wrap the reference schema in a single-case allOf:
            result = {"allOf": [json_schema]}
        else:
            result = json_schema

        if default is not None:
            result["default"] = default

        return result



# keep_lazy seems not needed as .title forces translation anyway
# https://github.com/vitalik/django-ninja/issues/774
# @keep_lazy_text
def title_if_lower(s: str) -> str:
    if s == s.lower():
        return s.title()
    return s


class AnyObject:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[..., Any]
    ) -> Any:
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: Any, handler: Callable[..., Any]
    ) -> DictStrAny:
        return {"type": "object"}

    @classmethod
    def validate(cls, value: Any, _: Any) -> Any:
        return value


TYPES = {
    "AutoField": int,
    "BigAutoField": int,
    "BigIntegerField": int,
    "BinaryField": bytes,
    "BooleanField": bool,
    "CharField": str,
    "DateField": datetime.date,
    "DateTimeField": datetime.datetime,
    "DecimalField": Decimal,
    "DurationField": datetime.timedelta,
    "FileField": str,
    "FilePathField": str,
    "FloatField": float,
    "GenericIPAddressField": IPvAnyAddress,
    "IPAddressField": IPvAnyAddress,
    "IntegerField": int,
    "JSONField": AnyObject,
    "NullBooleanField": bool,
    "PositiveBigIntegerField": int,
    "PositiveIntegerField": int,
    "PositiveSmallIntegerField": int,
    "SlugField": str,
    "SmallAutoField": int,
    "SmallIntegerField": int,
    "TextField": str,
    "TimeField": datetime.time,
    "UUIDField": UUID,
    # postgres fields:
    "ArrayField": List,
    "CICharField": str,
    "CIEmailField": str,
    "CITextField": str,
    "HStoreField": Dict,
}

TModel = TypeVar("TModel")


@no_type_check
def create_m2m_link_type(type_: Type[TModel]) -> Type[TModel]:
    class M2MLink(type_):  # type: ignore
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return core_schema.with_info_plain_validator_function(cls._validate)

        @classmethod
        def __get_pydantic_json_schema__(cls, schema, handler):
            json_type = {
                int: "integer",
                str: "string",
                float: "number",
                UUID: "string",
            }[type_]
            return {"type": json_type}

        @classmethod
        def _validate(cls, v: Any, _):
            try:
                return v.pk  # when we output queryset - we have db instances
            except AttributeError:
                return type_(v)  # when we read payloads we have primakey keys

    return M2MLink

class Schema(BaseModel, metaclass=ModelSchemaMetaclass):
    class Config:
        from_attributes = True  # aka orm_mode

    @model_validator(mode="wrap")
    @classmethod
    def _run_root_validator(
        cls, values: Any, handler: ModelWrapValidatorHandler[S], info: ValidationInfo
    ) -> Any:
        # If Pydantic intends to validate against the __dict__ of the immediate Schema
        # object, then we need to call `handler` directly on `values` before the conversion
        # to DjangoGetter, since any checks or modifications on DjangoGetter's __dict__
        # will not persist to the original object.
        forbids_extra = cls.model_config.get("extra") == "forbid"
        should_validate_assignment = cls.model_config.get("validate_assignment", False)
        if forbids_extra or should_validate_assignment:
            handler(values)

        values = DjangoGetter(values, cls, info.context)
        return handler(values)

    @classmethod
    def from_orm(cls: Type[S], obj: Any, **kw: Any) -> S:
        return cls.model_validate(obj, **kw)

    def dict(self, *a: Any, **kw: Any) -> DictStrAny:
        "Backward compatibility with pydantic 1.x"
        return self.model_dump(*a, **kw)

    @classmethod
    def json_schema(cls) -> DictStrAny:
        return cls.model_json_schema(schema_generator=NinjaGenerateJsonSchema)

    @classmethod
    def schema(cls) -> DictStrAny:  # type: ignore
        warnings.warn(
            ".schema() is deprecated, use .json_schema() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.json_schema()


SchemaKey = Tuple[Type[Model], str, int, str, str, str, str]


class SchemaFactory:
    def __init__(self) -> None:
        self.schemas: Dict[SchemaKey, Type[Schema]] = {}
        self.schema_names: Set[str] = set()

    def create_schema(
        self,
        model: Type[Model],
        *,
        name: str = "",
        depth: int = 0,
        fields: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None,
        custom_fields: Optional[List[Tuple[str, Any, Any]]] = None,
        base_class: Type[Schema] = Schema,
    ) -> Type[Schema]:
        schema: Type[Schema]
        name = name or model.__name__

        key = self.get_key(
            model, name, depth, fields, exclude, optional_fields, custom_fields
        )
        if schema := self.get_schema(key):
            return schema

        definitions = self.convert_django_fields(
            model,
            depth=depth,
            fields=fields,
            exclude=exclude,
            optional_fields=optional_fields,
        )
        
        if custom_fields:
            for fld_name, python_type, field_info in custom_fields:
                # if not isinstance(field_info, FieldInfo):
                #     field_info = Field(field_info)
                definitions[fld_name] = (python_type, field_info)
        
        schema = create_pydantic_model(
            name,
            __config__=None,
            __base__=base_class,
            __module__=base_class.__module__,
            __validators__={},
            **definitions,
        )  # type: ignore

        self.add_schema(key, name, schema)
        return schema

    def get_schema(self, key: SchemaKey) -> Union[Type[Schema], None]:
        if key in self.schemas:
            return self.schemas[key]
        return None

    def add_schema(self, key: SchemaKey, name: str, schema: Type[Schema]):
        if name in self.schema_names:
            name = self._get_unique_name(name)
        self.schemas[key] = schema
        self.schema_names.add(name)

    def convert_django_fields(
        self,
        model: Type[Model],
        *,
        depth: int = 0,
        fields: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Any, Any]]:
        if fields and exclude:
            raise ConfigError("Only one of 'fields' or 'exclude' should be set.")

        model_fields_list = list(self._selected_model_fields(model, fields, exclude))
        if optional_fields:
            if optional_fields == "__all__":
                optional_fields = [f.name for f in model_fields_list]

        definitions = {}
        for fld in model_fields_list:
            python_type, field_info = get_schema_field(
                fld,
                depth=depth,
                optional=optional_fields and (fld.name in optional_fields),
            )
            definitions[fld.name] = (python_type, field_info)

        return definitions

    def get_key(
        self,
        model: Type[Model],
        name: str,
        depth: int,
        fields: Union[str, List[str], None],
        exclude: Optional[List[str]],
        optional_fields: Optional[Union[List[str], str]],
        custom_fields: Optional[List[Tuple[str, str, Any]]],
    ) -> SchemaKey:
        "returns a hashable value for all given parameters"
        # TODO: must be a test that compares all kwargs from init to get_key
        return (
            model,
            name,
            depth,
            str(fields),
            str(exclude),
            str(optional_fields),
            str(custom_fields),
        )

    def _get_unique_name(self, name: str) -> str:
        "Returns a unique name by adding counter suffix"
        for num in itertools.count(start=2):  # pragma: no branch
            result = f"{name}{num}"
            if result not in self.schema_names:
                break
        return result

    def _selected_model_fields(
        self,
        model: Type[Model],
        fields: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Iterator[DjangoField]:
        "Returns iterator for model fields based on `exclude` or `fields` arguments"
        all_fields = {f.name: f for f in self._model_fields(model)}

        if not fields and not exclude:
            for f in all_fields.values():
                yield f

        invalid_fields = (set(fields or []) | set(exclude or [])) - all_fields.keys()
        if invalid_fields:
            raise ConfigError(
                f"DjangoField(s) {invalid_fields} are not in model {model}"
            )

        if fields:
            for name in fields:
                yield all_fields[name]
        if exclude:
            for f in all_fields.values():
                if f.name not in exclude:
                    yield f

    def _model_fields(self, model: Type[Model]) -> Iterator[DjangoField]:
        "returns iterator with all the fields that can be part of schema"
        for fld in model._meta.get_fields():
            if isinstance(fld, (ManyToOneRel, ManyToManyRel)):
                # skipping relations
                continue
            yield cast(DjangoField, fld)


factory = SchemaFactory()

create_schema = factory.create_schema





@no_type_check
def get_schema_field(
    field: DjangoField, *, depth: int = 0, optional: bool = False
) -> Tuple:
    "Returns pydantic field from django's model field"
    alias = None
    default = ...
    default_factory = None
    description = None
    title = None
    max_length = None
    nullable = False
    python_type = None

    if field.is_relation:
        if depth > 0:
            return get_related_field_schema(field, depth=depth)

        internal_type = field.related_model._meta.pk.get_internal_type()

        if not field.concrete and field.auto_created or field.null or optional:
            default = None
            nullable = True

        alias = getattr(field, "get_attname", None) and field.get_attname()

        pk_type = TYPES.get(internal_type, int)
        if field.one_to_many or field.many_to_many:
            m2m_type = create_m2m_link_type(pk_type)
            python_type = List[m2m_type]  # type: ignore
        else:
            python_type = pk_type

    else:
        _f_name, _f_path, _f_pos, field_options = field.deconstruct()
        blank = field_options.get("blank", False)
        null = field_options.get("null", False)
        max_length = field_options.get("max_length")

        internal_type = field.get_internal_type()
        python_type = TYPES[internal_type]

        if field.primary_key or blank or null or optional:
            default = None
            nullable = True

        if field.has_default():
            if callable(field.default):
                default_factory = field.default
            else:
                default = field.default

    if default_factory:
        default = PydanticUndefined

    if nullable:
        python_type = Union[python_type, None]  # aka Optional in 3.7+

    description = field.help_text or None
    title = title_if_lower(field.verbose_name)

    return (
        python_type,
        FieldInfo(
            default=default,
            alias=alias,
            validation_alias=alias,
            serialization_alias=alias,
            default_factory=default_factory,
            title=title,
            description=description,
            max_length=max_length,
        ),
    )


@no_type_check
def get_related_field_schema(field: DjangoField, *, depth: int) -> Tuple["OpenAPISchema"]:
    from ninja.schema import create_schema

    model = field.related_model
    schema = create_schema(model, depth=depth - 1)
    default = ...
    if not field.concrete and field.auto_created or field.null:
        default = None
    if isinstance(field, ManyToManyField):
        schema = List[schema]  # type: ignore

    return (
        schema,
        FieldInfo(
            default=default,
            description=field.help_text,
            title=title_if_lower(field.verbose_name),
        ),
    )
