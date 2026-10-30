"""Online app with the same image budget used for the measured pilot."""
from .app import Settings,create_app
from .demand_model import DemandModel

def create_online_app():
    settings=Settings()
    return create_app(settings,DemandModel(settings.model_name,settings.revision))
