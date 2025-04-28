# Changelog
Notable changes to `nfdata` will be documented here. We are using [semantic versioning](https://semver.org/), though note pre-release versions (0.0.x) may include breaking changes.


## [Unreleased]

## [0.3.0] - 2025-04-25

### Added
* Ability to calculate flow direction from a conditioned DEM. If a `dem` raster is provided and no `flow_dir` raster is, the DEM will be used to calculate the flow direction.
* Ability to attempt to resolve an unconditioned DEM (with pits, depressions and flats) as part of the flow direction calculations, using Whitebox Tools `breach_depressions_least_cost`. New config options added accordingly (under key `condition_dem`).
* Beginings of a test suite (see `tests/`), though there is a long way to go until a suite of unit tests with full coverage.

### Changed
* The `Compiler` initialisation routine now uses the `model_vars.yaml` dict included in package data as default, with the `model_vars_path` parameter being available to override this for a custom model vars dict.
* Reprocessed the Thames TiO2 2015 example data DEM from an ASCII grid to a TIFF (`data.example/thames_tio2_2015/dem.tif`), and clipped to the correct extent so that it can be used to calculate flow direction from.

### Fixed
* General code readibility improvements and moving towards PEP8 compliance (not there yet).
* Refactored routing into an non-class-based utility-type module `routing.py`.


## [0.2.1] - 2025-03-05

* No changes, just fixes to CI/CD.


## [0.2.0] - 2025-03-05

* Implemented CI/CD. Package is now uploaded to PyPI and Anaconda on every release.


## [0.1.0] - 2024-07-06

* Initial version.


[Unreleased]: https://github.com/nerc-ceh/nanofase-data/compare/0.3.0...HEAD
[0.3.0]: https://github.com/nerc-ceh/nanofase-data/releases/tag/0.3.0
[0.2.1]: https://github.com/nerc-ceh/nanofase-data/releases/tag/0.2.1
[0.2.0]: https://github.com/nerc-ceh/nanofase-data/releases/tag/0.2.0
[0.1.0]: https://github.com/nerc-ceh/nanofase-data/releases/tag/0.1.0