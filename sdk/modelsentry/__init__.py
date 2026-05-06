"""ModelSentry SDK — early warning when production ML models degrade.

Statistical profiles are computed locally; raw feature values and raw
predictions never leave the customer environment.

    import modelsentry as ms

    ms.init(api_key="...", model_id="churn-v3")

    @ms.monitor()
    def predict(features):
        return model.predict(features)
"""
from modelsentry.drift import DriftReport
from modelsentry.monitor import init, monitor
from modelsentry.profiler import Profile

__version__ = "0.1.0"
__all__ = ["DriftReport", "Profile", "__version__", "init", "monitor"]
