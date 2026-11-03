# Import-Export – Lösung starten und prüfen

Die fertige Abgabe liegt in `artifacts/Import-Export.csv`. Das technische Dokument
für das Formular ist `Import-Export.md`. Der Quellcode liegt unter `solution/`.

## Kurz erklärt

1. Das Programm liest ausschließlich das **lokale taktische Tracking** und das
   **automatische Smart Tagging**. Es lädt keine Spieldaten ins Internet.
2. Es bewertet, welches Team den Ball kontrolliert: Abstand zum Ball, Balltempo,
   konkurrierende Spieler und automatische Angriffs-/Verteidigungsphasen.
3. Es sucht Hinweise auf Unterbrechungen: Ball außerhalb des Felds, ruhender Ball
   zusammen mit wenig Spielerbewegung und automatische Standardsituationen.
4. Ein Zustandsmodell verbindet die Hinweise über die Zeit. Ein kurzer Abpraller
   soll keinen neuen Ballbesitz erzeugen. Ein unsichtbarer Ball ist nicht
   automatisch eine Spielunterbrechung.
5. Es schreibt eine Zeile pro Videobild und prüft das offizielle Abgabeformat.

Die Parameter sind feste, nicht trainierte Entwicklungsannahmen. Es wurde weder
auf diesem Spiel noch auf seiner Ground Truth oder seinem Video trainiert oder
getunt. Die neuere Regel-E-Mail hat Vorrang vor älteren Aussagen in diesem Repo.

## Auf diesem Mac ausführen

Im Terminal:

```bash
cd /Users/lukajuric/Desktop/more-than-a-hack-2027
.venv/bin/python -m solution.infer --intermediates
```

Die virtuelle Python-Umgebung wurde bereits eingerichtet. Der Lauf funktioniert
anschließend offline. Die Videos werden nicht gebraucht; deren abweichende
Dateinamen mit `-001`/`-002` stören daher nicht.

Alternativ im Finder `run_solution.command` doppelklicken. Das Skript installiert
die festgeschriebenen Python-Pakete nur, falls die Umgebung noch fehlt.

## Testen

```bash
# Künstliche Software-Testfälle, ohne Wettbewerbsdaten:
.venv/bin/python -m unittest discover -s tests -v

# Abgabeformat, Halbzeitgrenzen und Prüfsummen:
.venv/bin/python -m solution.verify

# Zusätzlich alles erneut berechnen und auf bytegleiche Ausgabe prüfen:
.venv/bin/python -m solution.verify --reproduce
```

Der letzte Test sperrt Netzwerkzugriffe sowie das Lesen von Videos und
Ground-Truth-Dateien während der erneuten Berechnung. Ein `PASS` beweist die
jeweilige technische Eigenschaft, **nicht die sportliche Genauigkeit**.

Der unveränderte offizielle Formatprüfer funktioniert ebenfalls:

```bash
.venv/bin/python evaluation/validate_submission.py artifacts/Import-Export.csv
```

## Einmaliger Check auf den öffentlichen fünf Minuten

Für die fertiggestellte, eingefrorene Version wurde der offizielle öffentliche
Evaluator einmal erfolgreich ausgeführt. Der Bericht bleibt lokal unter
`artifacts/public_check/report.txt`; die zum Check gehörenden Prüfsummen unter
`artifacts/public_check/frozen_manifest.json`. Weder die Parameter noch die
Vorhersagen wurden anhand des Berichts verändert. Der Bericht wurde nicht zur
KI-gestützten Analyse eingelesen.

Die Wettbewerbsunterlagen erlauben den öffentlichen Ausschnitt zur technischen
Pipeline-Prüfung. Er ist **kein Tuning- oder Auswahlset**. Die Lösung wird nicht
automatisch gegen die Ground Truth optimiert und liest sie bei der Berechnung nie.

Für einen getrennten lokalen Bericht nach Fertigstellung:

```bash
.venv/bin/python evaluation/run_evaluation.py --results artifacts --gt public --out artifacts/public_check
```

Keine Parameter oder Varianten anhand dieses Ergebnisses auswählen. Ausgabe
niemals mit dem Video abgleichen. Verbesserungen, die Training oder Tuning
benötigen, müssen ausschließlich auf anderen, passend lizenzierten Spielen
entwickelt werden. Die fünf Minuten sagen nicht zuverlässig voraus, ob wir
weiterkommen. Eine bessere Gesamtspielwertung als die Baseline ist bislang
**nicht nachgewiesen**.

## Auf einem anderen Rechner reproduzieren

Getestete Umgebung: **Python 3.14.3**, Pakete aus `requirements-solution.txt`.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-solution.txt
.venv/bin/python -m solution.infer --data-dir /pfad/zum/datenordner
.venv/bin/python -m solution.verify --data-dir /pfad/zum/datenordner --reproduce
```

Der Datenordner muss `tactical/BAR-ATH_tactical_atd.xml`,
`tactical/BAR-ATH_tactical_atd.txt` und
`tactical/BAR-ATH_tactical_smart_tagging.xml` enthalten. Alternativ funktioniert
die Umgebungsvariable `MTAH_DATA_DIR`.

## Dateien und Abgabe

| Datei | Zweck |
|---|---|
| `artifacts/Import-Export.csv` | Diese Datei im Abgabeformular hochladen |
| `Import-Export.md` | Technisches Dokument für das Abgabeformular |
| `solution/`, `tests/`, `requirements-solution.txt` | Eigener reproduzierbarer Code |
| `artifacts/Import-Export.manifest.json` | Prüfsummen der Eingaben, des Codes und der Ausgabe |
| `artifacts/Import-Export.half*.npz` | Optionale lokale Zwischenergebnisse |
| `artifacts/Import-Export.restarts.json` | Automatisch geschätzte Wiederanstoßgrenzen |
| `artifacts/public_check/report.txt` | Einmaliger lokaler Bericht des offiziellen Evaluators |
| `dist/Import-Export-code.zip` | Quellcodepaket ohne Spiel- und Diagnosedaten |

Im Formular den Teamnamen **Import-Export** und den Link auf euer öffentliches
Code-Repository eintragen. Der neue Code muss dort vor der Abgabe verfügbar sein;
lokale Dateien allein aktualisieren GitHub nicht. Die Formular-URL kommt laut
`SUBMISSION.md` per E-Mail. Die Deadline dort ist **17. November 2026**.

`data/`, `artifacts/`, `dist/` und `.venv/` bleiben gitignoriert. Die Abgabe-CSV
gehört ins Formular, die Spieldaten und lokalen Diagnose-Dateien nicht in ein
öffentliches Repository. Keine manuelle Korrektur der Abgabe-CSV.

Das Quellcodepaket kann mit `.venv/bin/python -m solution.package` erneut erstellt
werden. Es enthält eine feste Auswahl an Code, Tests und Dokumentation, keine
Spieldaten oder Ground-Truth-Dateien. Für den optionalen öffentlichen Score-Check
im entpackten Paket muss die öffentliche Ground Truth separat aus dem offiziellen
Wettbewerbsrepository unter `evaluation/gt/` bereitgestellt werden.
