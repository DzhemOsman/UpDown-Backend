class DataSourceError(Exception):
    """
    Externe Datenquelle (InfluxDB oder Yahoo Finance) nicht erreichbar
    oder fehlerhaft. Signalisiert ein Infrastruktur-Problem - bewusst
    abgegrenzt von 'Ticker hat schlicht keine Daten'.
    """
