# How to contribute

## Release new version

1. Change the version number in 

- pylocus/__init__.py
- setup.py
- docs/source/conf.py 

2. Add a new section to CHANGELOG.md

3. Run the following commands

```bash
  python3 setup.py sdist bdist_wheel
  python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps pylocus 
  python3 -c "import pylocus.lateration" # test that it worked.
```

4. If all looks ok, then run

```
python3 -m twine upload dist/*
```

(If above does not work, make sure that the information in ~/.pypirc is up to date)

5. Do not forget to create a new release on github as well.

## Update documentation on github

This is done automatically. If it fails, you can check your local installation using

```
pip install -r docs/requirements.txt
mkdir docs/build
sphinx-build -b html docs/source/ docs/build
```

And open `docs/build/index.html` to see the output.
