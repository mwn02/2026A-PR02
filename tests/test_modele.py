import numpy as np
import pandas as pd
import pytest

from modele import DummyModel, SGDModel
from sklearn.exceptions import NotFittedError


@pytest.fixture
def donnees():
    return pd.DataFrame({
        "Hour": [12] * 100,
        "Humidity(%)": [50] * 100,
        "Wind speed (m/s)": [2.0] * 100,
        "Temperature(°C)": [15.0] * 100,
        "Seasons": ["Winter", "Summer"] * 50,
        "Holiday": ["No Holiday"] * 100,
        "Rainfall(mm)": [0.0] * 100,
        "Rented Bike Count": [100, 900] * 50,
    })


@pytest.mark.parametrize("classe", [DummyModel, SGDModel])
def test_separation_donnees(classe, donnees):
    modele = classe(donnees)

    assert len(modele.x_train) == 80
    assert len(modele.x_test) == 20
    assert set(modele.x_train.index).isdisjoint(modele.x_test.index)


def test_dummy_predit_moyenne(donnees):
    modele = DummyModel(donnees)
    modele.entrainer()

    predictions = modele.predict(modele.x_test)

    np.testing.assert_allclose(predictions, modele.y_train.mean()) # test s'il retourne la moyenne

def test_not_fitted_model_error(donnees):
    modele = DummyModel(donnees)
    with pytest.raises(NotFittedError):
        modele.predict(donnees)