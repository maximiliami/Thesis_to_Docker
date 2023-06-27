# Thesis_Maximilian_Rupprecht

## Themenstellung
Im Internet der Dinge (loT) sammeln Millionen von Geräten wie Sensoren Daten und übermitteln diese an loT-Plattformen. Dort werden die Daten gespeichert und in einem ersten Schritt visualisiert. Um eine zuverlässige Datenerfassung sicherzustellen muss der Zustand der Sensoren überwacht und die Batterien rechtzeitig ausgetauscht werden.
Das Ziel dieser Arbeit besteht darin vorhandene Sensordaten aus der loT-Plattform ThingsBoard Professional Edition zu extrahieren, die Daten aufzubereiten und für
Prädiktionen bezüglich der verbleibenden Akku- bzw. Batterielaufzeit zu nutzen.
Dabei sollen geeignete Protokolle für die Extraktion untersucht werden und geeignete Algorithmen bestimmt werden, um die Funktionsfähigkeit eines akku- oder batteriebetriebenen Sensors vorherzusagen.
Die Prädiktion wird der loT-Plattform zurückgeführt und in einem Dashboard visualisiert. Dabei ist zu evaluieren, wie die Prädiktion in die loT-Plattform geeignet integriert wird.

## Struktur
    |-Thesis_to_Docker
        |-cron              beinhaltet den cronjob
        |-data              beinhaltet sämtliche Daten
        | |-model           Ordner für trainierte Modelle
        | |-png             Ordner für Bilddateien
        | |-scaler          Ordner für StandardScaler-Objekte
        | |-sensor_data     Ordner für Sensordaten
        | |-sensor_preview  Ordner für die automatisierte Sensorvorschau
        | | |-png           Ordner für Bilddateien der automatisierten Sensorvorschau
        | |-testdata        Ordner für Testdatensätze
        | |-trainingsdata   Ordner für Trainingsdatensätze
        |-thesis_code       beinhaltet sämtlichen Code und Jupyter-Notebooks
        | |-ARIMA           beinhaltet Jupyter-Notebooks zur Prädiktion der Spapnnungsverlaufsdaten mit einem ARIMA-Modell
        | |-data_extr...    beinhaltet Jupyter-Notebooks zur Extraktion und Analyse von Spannungsverlaufsdaten aus ThingsBoard
        | |-forecast        beinhaltet ein Pythonskript zur Prädiktion der aktuellen Spannungsverlaufsdaten der LGT92-23-33 Sensoren
        | |-LSTM            beinhaltet Jupyter-Notebooks zum Erstellen, Trainieren und Evalulieren von LSTM-Modellen
        | |-service         beinhaltet Service-Klassen
        | |-settings        beinhaltet Einstellungsmöglichkeiten !! Authentifizierungsdatei !!
        |
        |-Dockerfile_cron   Dockerfile zum Builden eines Dockercontainers, welcher einen Cronjob für das Ausführen des Forecast-Skript beinhaltet
        |-Dockerfile_for..  Dockerfile zum Builden eines Dockercontainers, welcher das Forecast-Skript beim Starten des Containers ausführt
        |-Dockerfile_jup..  Dockerfile zum Builden eines Dockercontainers, welcher beim Starten einen Jupyter-Server startet
        |-README.md         Informationen über das Projekt
        |-requirements.txt  Informationen über genutzte Packages

## Verwendung
Um das Skript zur Prädiktion der Spannungsverlaufsdaten sowie die Jupyter-Notebooks zur Erkundung der Funktionalität und Ergebnisse des Projekts auf verschiedenen Plattformen verfügbar zu machen, werden drei Dockerfiles bereitgestellt. Diese Dockerfiles ermöglichen die Initialisierung von Docker-Containern zur Ausführung und Nutzung der Skripte und Jupyter-Notebooks.

Es stehen folgende drei Dockerfiles zur Verfügung, welche unterschiedliche Funktionalitäten definieren, die in den Containern verfügbar gemacht werden:

1. Dockerfile_cron
   - Dieses Dockerfile ermöglicht die Erstellung eines Containers, der beim Start einen Cronjob ausführt. Der Cronjob ist so konfiguriert, dass das Skript zur automatisierten Prädiktion der Spannungsverlaufsdaten einmal täglich ausgeführt wird. Dadurch wird eine regelmäßige Aktualisierung der Prädiktionen ohne manuellen Eingriff gewährleistet.
2. Dockerfile_forecast
   - Dieses Dockerfile ermöglicht die Erstellung eines Containers, der beim Start das Skript zur Prädiktion der Spannungsverlaufsdaten ausführt und sich anschließend automatisch beendet. Dieser Container eignet sich ideal für den Einsatz in automatisierten Umgebungen. Er könnte beispielsweise über einen Cronjob gesteuert werden, um sicherzustellen, dass das System nicht durch einen permanent laufenden Container beeinträchtigt wird.
3. Dockerfile_jupyter
   - Mit diesem Dockerfile wird ein Jupyter-Notebook-Server erstellt, der eine plattformunabhängige Ausführung der Jupyter-Notebooks ermöglicht. Durch die Nutzung dieses Servers können die Notebooks in einer interaktiven Umgebung geöffnet werden, um den Code auszuführen, visuelle Darstellungen zu generieren und die Funktionalität sowie die Ergebnisse des Projekts zu erkunden.

## Nutzung der Docker-Container

**Überprüfen Sie die Ordnerstruktur nach dem GitHub-Download und entfernen Sie ggf. "-master" aus dem Verzeichnisnamen "Thesis_to_Docker-master" >> "Thesis_to_Docker" um die definierte Ordnerstruktur zu erhalten**

**!! In der Klasse „Authentication“ im Ordner Settings müssen URL, Username und Passwort zur Authentifizierung an der IoT-Plattform ThingsBoard hinterlegt werden, um alle Funktionalitäten nutzen zu können. !!**

Damit die Docker-Container bereitgestellt werden können, vergewissern Sie sich, dass Docker auf Ihrem System installiert ist. Öffnen Sie die Konsole und navigieren Sie in das Verzeich
nis, in dem die Dockerfiles liegen und führen Sie folgende dockerfile-spezifische Befehle aus:

1. Dockerfile_cron
   - Erstellen des Images:
     - ```docker build -t name_des_containers -f Dockerfile_cron .```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
   - Starten des Containers:
     - ```docker run -it name_des_containers```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
2. Dockerfile_forecast
   - Erstellen des Images:
     - ```docker build -t name_des_containers -f Dockerfile_forecast .```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
   - Starten des Containers:
     - ```docker run -it name_des_containers```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
3. Dockerfile_jupyter:
   - Erstellen des Images:
     - ```docker build -t name_des_containers -f Dockerfile_jupyter .```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
   - Starten des Containers und mappen der Ports:
     - ```docker run -p 8888:8888 -it name_des_containers```
     - Ersetzen Sie "name_des_containers" durch den gewünschten Namen für den Container.
   - In der Konsole wird ein Link angezeigt, unter dem der Jupyter-Notebook-Server zu erreichen ist. 

