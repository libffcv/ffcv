rm -rf _build/html/*
rm -rf api/*
ls "api/"
sphinx-apidoc -E -e -M -o api/ ../ffcv/
python reformat.py
make html