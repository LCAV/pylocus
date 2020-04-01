# Changelog

## Unreleased

- Documentation fixes

## [0.0.5] - 2019-11-08
### Changed

- Remove unused angle algorithms, keeping only basic angle operations. The angle-based algorithms are a niche application and are now in a different respository.

- Make cvxpy a compulsory dependence.


## [0.0.4] - 2019-09-12
### Changed

- Added more detailed documentation. 
- More flexibility in passing anchor coordinates: Before, one had to add n zero-vectors to the 
beginning of the coordinates matrix, one for each point to be localized. Now, one can pass
the anchor coordinates only, and the zero-vectors will be added automatically where needed.
- Cleaned up requirements.

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
