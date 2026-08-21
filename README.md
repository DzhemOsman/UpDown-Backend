# UpDown Backend

FastAPI-basiertes Backend für das Backtesting und die Optimierung von Mean-Reversion-Trading-Strategien unter Nutzung von InfluxDB 3 Core.

## Features

- **Marktdaten-Ingestion:** Automatisierter Abruf von OHLCV-Daten via Yahoo Finance in **InfluxDB 3 Core**.
- **Trading-Strategien:** Optimierte Mean-Reversion-Strategien mit SMA-Trendfiltern, Kadane-Algorithmus und dynamischem Money-Management.
- **Backtesting & Optimierung:** Grid Search und Bayes'sche Optimierung (`optuna`) zur Findung profitabler Parameter (Drop Threshold, Hold Days, Take Profit, Stop Loss).
- **REST API:** Bereitstellung per **FastAPI** und **Uvicorn** zur Anbindung an ein Frontend.

---

## Voraussetzungen

Da das Backend lokal ausgeführt wird und die InfluxDB in einem Container läuft, benötigst du Folgendes:
1. **Python 3.12+** (für das Backend)
2. **Docker Desktop** (für die InfluxDB)

### Installation von Docker Desktop (Windows)
1. Lade [Docker Desktop für Windows](https://docs.docker.com/desktop/install/windows-install/) herunter.
2. Führe die heruntergeladene `.exe`-Datei aus. Stelle sicher, dass bei der Installation **WSL 2** als Backend ausgewählt ist (empfohlen).
3. Starte deinen Computer nach Aufforderung neu.
4. Öffne Docker Desktop, akzeptiere die Nutzungsbedingungen und warte, bis der Status unten links auf "Engine running" springt.

### Installation von Docker Desktop (macOS)
1. Lade [Docker Desktop für Mac](https://docs.docker.com/desktop/install/mac-install/) herunter. Wähle die passende Version für deinen Mac (Apple Silicon oder Intel-Chip).
2. Öffne die `.dmg`-Datei und ziehe das Docker-Icon in den `Applications`-Ordner (Programme).
3. Starte Docker aus deinen Anwendungen.
4. Gib dein Systempasswort ein, wenn du nach Berechtigungen gefragt wirst, und warte, bis die Docker-Engine läuft.

---

## InfluxDB 3 Core Setup (via Docker Desktop)

Wir starten und konfigurieren die InfluxDB direkt über die grafische Benutzeroberfläche von Docker Desktop.

**1. Image herunterladen & starten:**
1. Öffne Docker Desktop und klicke oben in die Suchleiste.
2. Suche nach `influxdb:3-core` und wähle das offizielle Image aus.
3. Klicke auf **Pull** (oder direkt auf **Run**, falls verfügbar), um das Image herunterzuladen.
4. Gehe auf der linken Seite zum Menüpunkt **Images**.
5. Suche in der Liste nach `influxdb` (Tag: `3-core`) und klicke auf den Play-Button (**Run**).

**2. Container konfigurieren:**
Es öffnet sich ein Fenster "Run a new container". Klicke auf **Optional settings** (Optionale Einstellungen ausklappen) und trage Folgendes ein:
* **Container name:** `influxdb`
* **Ports:** Trage bei "Host port" `8086` ein (der Container Port ist ebenfalls 8086).
* **Volumes:** * Host path: (leer lassen oder einen lokalen Ordner auswählen, z.B. `C:\influxdata` bzw. `~/influxdata`)
  * Container path: `/var/lib/influxdb3`
* **Environment variables:** * Variable: `INFLUXDB3_NODE_IDENTIFIER_PREFIX`
  * Value: `updown`
* Klicke anschließend auf **Run**.

**3. Admin-Token generieren:**
1. Gehe im linken Menü auf **Containers**.
2. Klicke auf den Namen deines laufenden Containers (`influxdb`).
3. Wechsel oben auf den Reiter **Exec** (das ist das integrierte Terminal des Containers).
4. Tippe folgenden Befehl ein und drücke Enter:
   ```bash
   influxdb3 create token --admin
WICHTIG: Markiere und kopiere das angezeigte Token! Es wird aus Sicherheitsgründen nur dieses eine Mal angezeigt.

4. Datenbank erstellen:
Tippe nun im selben Exec-Terminal den folgenden Befehl ein, um die Datenbank updown zu erzeugen (ersetze <DEIN_TOKEN> durch das gerade kopierte Token) und drücke Enter:

```bash
    influxdb3 create database updown --token <DEIN_TOKEN>
```
Quickstart (Applikation lokal starten)
Sobald die InfluxDB läuft, kannst du das Python-Backend direkt auf deinem System starten:

1. Umgebungsvariablen konfigurieren:
Erstelle eine .env-Datei aus der Vorlage im Projektverzeichnis.

Für macOS / Linux / Windows PowerShell:

```bash
cp .env.example .env
```
Für Windows (CMD / Eingabeaufforderung):


```DOS
copy .env.example .env
```
Öffne anschließend die neu erstellte .env-Datei in einem Texteditor und trage dein zuvor kopiertes InfluxDB Admin-Token bei der Variable INFLUXDB_TOKEN ein.

2. Abhängigkeiten installieren:
Installiere die benötigten Python-Bibliotheken (am besten innerhalb einer virtuellen Umgebung):

```Bash
pip install -r requirements.txt
```
3. Backend starten:
Starte den Entwicklungsserver mit Uvicorn:

```Bash
uvicorn app.main:app --reload --app-dir src
```
Das Backend ist nun unter http://localhost:8000 erreichbar. Die interaktive API-Dokumentation (Swagger UI) findest du unter http://localhost:8000/docs.

---

## ML-Modelle lokal regenerieren

Die trainierten Modelle (`models/*.pkl`, `*.pth`, `*.json`) **sind Teil des Git-Repos**. Grund: Training kostet auf der Produktions-VM unnötig Rechenzeit/Geld, daher wird lokal trainiert und das fertige Modell einfach mit committed - die VM trainiert nie selbst, sondern lädt die fertigen Artefakte per `git pull` und bindet `models/` per Volume in den Container ein (siehe `docker-compose.prod.yml`).

Reihenfolge zum (Re-)Trainieren:

1. **Diversifiziertes Ticker-Universum + Marktkontext in InfluxDB laden** (S&P 500 + DAX, ^GSPC, ^VIX - einmalig bzw. bei Bedarf erneut, dauert je nach Internetverbindung mehrere Minuten):
   ```bash
   $env:PYTHONPATH="src"
   python -m app.ml.bulk_ingest
   ```
2. **LightGBM trainieren** (Walk-Forward-Validierung über mehrere Marktregime + finales Produktivmodell + `models/lgbm_trend_diversified.meta.json` mit datengetriebenem Inferenz-Threshold):
   ```bash
   $env:PYTHONPATH="src"
   python -m app.ml.train_lgbm
   ```
3. **Optional: LSTM trainieren** (aktuell noch mit einem statischen 80/20-Split, Walk-Forward folgt später nach demselben Muster wie LightGBM):
   ```bash
   $env:PYTHONPATH="src"
   python -m app.ml.train
   ```
4. **Neue Modell-Dateien committen und nach `master` pushen** (bzw. per Pull Request). Nach dem Deploy reicht auf der VM ein Container-Neustart (`docker compose -f docker-compose.prod.yml restart api`), damit die neuen Artefakte geladen werden - kein Rebuild und kein Training auf der VM nötig.

Empfehlung: Retraining alle paar Monate wiederholen, damit die Modelle nicht auf einem veralteten Marktstand "einfrieren" (siehe `recommended_threshold`/`data_end` in der jeweiligen `*.meta.json` sowie die Staleness-Warnung im Backend-Log, wenn ein Backtest-Zeitraum deutlich über das Trainingsende hinausgeht).