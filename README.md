# MediaShield

Image similarity indexing and search using CLIP and ISC features, with a simple fallback mode.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
```

## Build Index

Default (CLIP + ISC):

```bash
python build_index.py
```

ISC only:

```bash
python build_index.py --isc-only
```


## Search

```bash
python search.py images/af.png
```

ISC only:

```bash
python search.py images/af.png --isc-only
```
