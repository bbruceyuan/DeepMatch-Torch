rm -r dist/
python setup.py sdist
python setup.py bdist_wheel
twine check dist/*
twine upload dist/*