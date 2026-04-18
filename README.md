# Code der Masterarbeit
Dieses repo ist der Code für die Masterarbeit "Bewertung der Robustheit von
CNN-Architekturen gegenüber
Modellquantisierung mittels Methoden der
erklarbaren KI" von Cornelius Rottmair

## Git LSF
Die in dieser Arbeit trainierten und untersuchten Modelle sind aufgrund ihrer Dateigröße über Git LFS ausgelagert. Falls die Modelle nicht selbst trainiert, sondern lediglich geladen werden sollen, kann dies nach dem Klonen des Repositories über die folgenden Befehle erfolgen:

```bash
git lfs install
git lfs pull
```

## Python Environment

Um die beiden Pipelines auszuführen, müssen die benötigten Python-Pakete installiert sein. Dazu kann man entweder die Pakete direkt installieren oder eine isolierte Python Umgebung verwenden. Ein Werkzeug dafür ist Micromamba, das die Installation und Verwaltung der Pakete vereinfacht.

### Micromamba Installation

Micromamba kann schnell installiert werden mit:

```bash
curl -L https://micro.mamba.pm/install.sh | bash  
```
Mit Micromamba installiert (oder irgendwas Condaähnlichem) lassen sich die packete instalieren und aktivieren mit:

```bash
micromamba create -f environment.yml
micromamba activate Master
```


### Daten Dowload
Zum Trainieren der Modelle wird der in der Arbeit besprochene Datensatz benutzt. Um sicherzustellen, dass er am richtigen Ort ist, kann das Skript `setup_data.py` benutzt werden:

```bash
python setup_data.py
```

## Quantisierungs Pipeline

Wie in der Arbeit beschrieben, dient die erste Pipeline der Erzeugung verschiedener quantisierter Modellvarianten ausgehend von einem Basismodell.  
Dabei wird entweder ein bereits vorhandenes FP32-Modell geladen oder ein neues Modell initialisiert und feinjustiert.

Die vier erzeugten Modellvarianten sind: Post-Training-Quantisierung (PTQ), Quantization-Aware Training (QAT) sowie reduzierte Präzisionsformate in FP16 und BF16. Alle erzeugten Modelle werden automatisch gespeichert und optional hinsichtlich Genauigkeit und Laufzeit evaluiert.

Die Pipeline kann beispielsweise ausgeführt werden mit:

```bash
python quantize_pipeline.py \
  --model_type resnet50 \
  --root data/imagenet100/versions/8 \
  --epochs_ft 10 \
  --epochs_qat 5
```

Der Parameter `--model_type` bestimmt die verwendete Architektur. Architekturen in dieser Arbeit:

`vgg16`, `resnet50`, `mobilenet_v2`, `googlenet`, `densenet121`, `mnasnet`

Der Parameter `--root` verweist auf das zugrunde liegende Datenset und wird relativ zum Projektverzeichnis angegeben.

Über `--epochs_ft` wird gesteuert, wie lange das Basismodell initial feinjustiert wird, während `--epochs_qat` die Dauer des quantisierungsbewussten Trainings sowie der anschließenden Feinjustierung festlegt.

Optional kann über `--resume` ein bereits trainiertes Modell geladen werden, wodurch der initiale Trainingsschritt entfällt.  
Mit `--force` lassen sich bestehende Modellartefakte überschreiben, während `--skip_eval` die abschließende Evaluation deaktiviert.

Alle erzeugten Modelle sowie ein Performance-Report werden im jeweiligen Unterordner von `saved_models/<model_type>/` gespeichert.

## Vergleichs Pipeline

Die zweite Pipeline baut direkt auf den zuvor erzeugten Modellen auf und untersucht deren Verhalten im Kontext erklärbarer KI-Methoden.

Hierzu werden verschiedene XAI-Methoden auf eine definierte Anzahl von Validierungsbildern angewendet. Die resultierenden Erklärungen werden anschließend miteinander verglichen und in Form eines Konsistenzberichts zusammengefasst.

Ein typischer Aufruf dieser Pipeline ist:

```bash
python compare_pipeline.py \
  --model_type resnet50 \
  --root data/imagenet100/versions/8 \
  --img_count 1000 \
  --methods scorecam eigencam lime shap umap \
  --batch_size 64 \
  --umap_samples 500 \
  --save_dir saved_models
```

Der Parameter `--model_type` bestimmt die Architektur und muss identisch zur zuvor verwendeten Quantisierungs-Pipeline sein.

Der Parameter `--root` verweist auf das zugrunde liegende Datenset im Projektverzeichnis.

Der Parameter `--img_count` legt fest, wie viele Bilder aus dem Validierungsdatensatz für die Analyse verwendet werden.

Der Parameter `--batch_size` steuert die Batchgröße während der Verarbeitung der Bilder.

Der Parameter `--umap_samples` legt fest, wie viele Samples für die UMAP-basierte Analyse verwendet werden (wenn wegelassen wird er geamte Datensaz benutzt).

Der Parameter `--save_dir` bestimmt das Verzeichnis, in dem die Modellartefakte und Ergebnisse gespeichert werden.

Über `--methods` wird festgelegt, welche XAI-Verfahren angewendet werden. Methioden in dieser Arbeit:
 `Score-CAM`, `Eigen-CAM`, `LIME`, `SHAP` ,`UMAP`

Die Pipeline erzeugt abschließend einen Konsistenzbericht, der im jeweiligen Modellverzeichnis innerhalb von `saved_models/<model_type>/` gespeichert wird.


## Notebooks
Die drei Notebooks Quantize, XAI und Compare erläutern in dieser Reihenfolge die grundlegenden Überlegungen hinter den Pipelines, während das Notebook plotsmaco die in der Arbeit verwendeten Abbildungen erzeug.