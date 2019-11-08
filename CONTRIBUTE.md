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
  python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-pkg-your-username
  python3 -c "import pylocus.lateration" # test that it worked.
```

4. If all looks ok, then run

```
python3 -m twine upload dist/*
```

5. If above does not work, make sure that the information in ~/.pypirc is up to date.
