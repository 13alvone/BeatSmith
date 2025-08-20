import random

from hypothesis import HealthCheck, given, strategies as st, settings

from beatsmith.providers import internet_archive as ia


class DummyResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._data

    def close(self):
        pass


@given(
    docs=st.lists(
        st.dictionaries(
            st.text(),
            st.one_of(st.text(), st.integers(), st.none()),
            max_size=4,
        ),
        max_size=5,
    )
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_ia_search_random_fuzz(docs, monkeypatch):
    def fake_get(*args, **kwargs):
        return DummyResp({"response": {"docs": docs}})

    monkeypatch.setattr(ia, "_get_with_retry", fake_get)
    results = ia.ia_search_random(random.Random(0), rows=5, query_bias=None, allow_tokens=[], strict=False)
    assert isinstance(results, list)
