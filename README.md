# Vásárlói elégedettség előrejelzése Yelp-értékelésekből

Féléves feladat – *A mesterséges intelligencia alkalmazási területei* (Óbudai Egyetem, MSc Data Science, 2026/27/1)

Az értékelés szövegéből a csillagszámot (1–5★), illetve egyszerűsített változatban a szentimentet
(negatív / semleges / pozitív) jósoljuk meg. Egy finomhangolt **DistilBERT** modellt hasonlítunk össze egy
**TF-IDF + logisztikus regresszió** baseline-nal.

## Adat

[`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) – kiegyensúlyozott, 20 000 elemű
véletlen minta (csillagonként 4 000), rétegzett felbontás: 14 400 tanító / 1 600 validációs / 4 000 teszt.
3 osztályos címke: 1–2★ negatív, 3★ semleges, 4–5★ pozitív.

## Eredmények (teszthalmaz, 4 000 értékelés)

| Feladat | Modell | Accuracy | Macro-F1 |
|---|---|---|---|
| 5 osztály | Többségi osztály | 20.0% | 0.067 |
| 5 osztály | TF-IDF + logisztikus regresszió | 56.1% | 0.559 |
| 5 osztály | DistilBERT (256 token, 2 epoch) | 60.5% | 0.606 |
| 5 osztály | **DistilBERT (512 token, 3 epoch)** | **61.4%** | **0.613** |
| 3 osztály | Többségi osztály | 40.0% | 0.190 |
| 3 osztály | TF-IDF + logisztikus regresszió | 77.2% | 0.710 |
| 3 osztály | DistilBERT (256 token, 2 epoch) | 79.4% | 0.745 |
| 3 osztály | **DistilBERT (512 token, 3 epoch)** | **80.3%** | **0.755** |

A teljes táblázat: [`results/summary.csv`](results/summary.csv), ábrák: [`results/figures/`](results/figures/).

## Notebookok

| Notebook | Tartalom |
|---|---|
| `01_data_and_baseline.ipynb` | Mintavétel, felbontás, TF-IDF + logisztikus regresszió baseline |
| `02_distilbert.ipynb` | DistilBERT finomhangolás, max. 256 token, 2 epoch |
| `03_distilbert_512_3ep.ipynb` | DistilBERT, max. 512 token, 3 epoch; összevetés a 02-vel |
| `04_error_analysis.ipynb` | Hibaelemzés: hibák mérete, kalibráció, nyelvi jelenségek, példák |

Közös kód: `common.py` (címkék, adatfelbontás, metrikák, ábrák), `train_utils.py` (DistilBERT tanítás és kiértékelés).

## Futtatás

Python 3.12, NVIDIA GPU (tesztelve: GTX 1660, 6 GB, fp32).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

A notebookokat sorrendben kell futtatni (01 → 04). Az adathalmaz és az előtanított modell a Hugging Face-ről
töltődik le. Tanítási idő a GTX 1660-on: kb. 9 perc (02) és 17 perc (03) feladatonként.

A `data/`, `models/` és a teszthalmazra adott előrejelzések (`results/predictions/`) nincsenek a repóban
(méret, illetve a Yelp-értékelések szövegének továbbadhatósága miatt); a notebookok futtatáskor újra létrehozzák őket.
