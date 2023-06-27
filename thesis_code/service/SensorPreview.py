from settings.Authentication import Authentication
from DataService import DataService

# Previews specific sensors from settings.Sensoren.cv
if __name__ == '__main__':
    auth = Authentication()
    ds = DataService()
    ds.preview_from_csv(missing_values='none')
