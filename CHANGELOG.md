# Changelog

## Unreleased 

## [0.0.4] - 2019-01-28
### Changed

- Added more detailed documentation. 
- More flexibility in passing anchor coordinates: Before, one had to add n zero-vectors to the 
beginning of the coordinates matrix, one for each point to be localized. Now, one can pass
the anchor coordinates only, and the zero-vectors will be added automatically where needed.
- Code now runs without cvxpy installation (as long as it is not used)

## [0.0.3] - 2018-07-16
### Changed

- Made SRLS algorithm work more robustly and efficiently, all tests passing. 

## [0.0.2] - 2018-07-12
### Changed

- Changed all imports to absolute (e.g. .basics to pylocus.basics), so that pylocus can be used as a git submodule. 
- Added continuous integration. 

## [0.0.1] - 2018-04-20 
### Changed

- Updated requirements etc. for ICASSP release. 
