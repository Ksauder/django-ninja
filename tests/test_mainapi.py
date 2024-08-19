from ninja import NinjaAPI, Router
from ninja.security import APIKeyQuery
from ninja.testing import TestClient


class Auth(APIKeyQuery):
    def __init__(self, secret):
        self.secret = secret
        super().__init__()

    def authenticate(self, request, key):
        if key == self.secret:
            return key


api = NinjaAPI(auth=Auth("thestuff"))
router = Router()

client = TestClient(api)


@router.get("/test")
def get_router_basic(request):
    return {"msg": "hello"}


api.add_router("/router", router)


@api.get("/test")
def get_basic(request):
    return {"msg": "hello"}


@api.api_operation(["GET", "POST"], "/many")
def get_many(request):
    return {"msg": "get post"}


@api.get("/auth", auth=Auth("different"))
def get_auth(request):
    return {"msg": "get post"}


def test_dynamic_auth():
    import inspect

    print(inspect.getmembers(router.get))
    print(inspect.getmembers(router.post))
    assert client.get("/router/test?key=thestuff").status_code == 200
    api.auth[0].secret = "newstuff"
    assert client.get("/router/test?key=thestuff").status_code == 401
    assert client.get("/test?key=newstuff").status_code == 200

    assert client.get("/many?key=newstuff").status_code == 200
    assert client.post("/many?key=newstuff").status_code == 200
    api.auth[0].secret = "random"
    assert client.get("/many?key=newstuff").status_code == 401
    assert client.post("/many?key=newstuff").status_code == 401
    assert client.get("/many?key=random").status_code == 200

    assert client.get("/auth?key=different").status_code == 200
