# RocketPy Change Log

All notable changes to `RocketPy` project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- Types of changes:
    - `Added` for new features.
    - `Changed` for changes in existing functionality.
    - `Deprecated` for soon-to-be removed features.
    - `Removed` for now removed features.
    - `Fixed` for any bug fixes.
    - `Security` in case of vulnerabilities.

    Should not be here:
    - tests
    - github maintenance
    - merge commits

    Types of messages:
    - Usually the message is the PR title and number
    - If the PR is too long to accomplish all the changes (it shouldn't be...),
      you can use a second line to describe it

-->

## [Unreleased] - yyyy-mm-dd

<!-- These are the changes that were not released yet, please add them correctly.
Attention: The newest changes should be on top -->

### Added

- ENH: add animations for motor propellant mass and tank fluid volumes [#894](https://github.com/RocketPy-Team/RocketPy/pull/894)
- ENH: Add axial_acceleration attribute to the Flight class [#876](https://github.com/RocketPy-Team/RocketPy/pull/876)
- ENH: Rail button bending moments calculation in Flight class [#893](https://github.com/RocketPy-Team/RocketPy/pull/893)
- ENH: Built-in flight comparison tool (`FlightComparator`) to validate simulations against external data [#888](https://github.com/RocketPy-Team/RocketPy/pull/888)
- ENH: Add persistent caching for ThrustCurve API [#881](https://github.com/RocketPy-Team/RocketPy/pull/881)
- ENH: Compatibility with MERRA-2 atmosphere reanalysis files [#825](https://github.com/RocketPy-Team/RocketPy/pull/825)
- ENH: Enable only radial burning [#815](https://github.com/RocketPy-Team/RocketPy/pull/815)
- ENH: Add thrustcurve api integration to retrieve motor eng data [#870](https://github.com/RocketPy-Team/RocketPy/pull/870)
- ENH: custom warning no motor or aerosurface [#871](https://github.com/RocketPy-Team/RocketPy/pull/871)
- ENH: Implement Bootstrapping for Confidence Interval Estimation [#891](https://github.com/RocketPy-Team/RocketPy/pull/897)

### Changed

-

### Fixed

- BUG: Fix parallel Monte Carlo simulation showing incorrect iteration count [#806](https://github.com/RocketPy-Team/RocketPy/pull/806)
- BUG: Fix CSV column header spacing in FlightDataExporter [#864](https://github.com/RocketPy-Team/RocketPy/issues/864)


## [v1.11.0] - 2025-11-01

### Added

- ENH: Tank Fluids with Variable Density from Temperature and Pressure [#852](https://github.com/RocketPy-Team/RocketPy/pull/852)
- ENH: Controller (AirBrakes) and Sensors Encoding [#849](https://github.com/RocketPy-Team/RocketPy/pull/849)
- EHN: Addition of ensemble variable to ECMWF dictionaries [#842](https://github.com/RocketPy-Team/RocketPy/pull/842)
- ENH: Added Crop and Clip Methods to Function Class [#817](https://github.com/RocketPy-Team/RocketPy/pull/817)
- DOC: Add Flight class usage documentation and update index [#841](https://github.com/RocketPy-Team/RocketPy/pull/841)
- ENH: Discretized and No-Pickle Encoding Options [#827](https://github.com/RocketPy-Team/RocketPy/pull/827)
- ENH: Add the Coriolis Force to the Flight class [#799](https://github.com/RocketPy-Team/RocketPy/pull/799)
- ENH: Improve parachute geometric parametrization [#835](https://github.com/RocketPy-Team/RocketPy/pull/835)
- ENH: Changing ellipses plot axis label [#855](https://github.com/RocketPy-Team/RocketPy/pull/855)

### Changed

- REL: bumps up rocketpy version to 1.11.0 [#868](https://github.com/RocketPy-Team/RocketPy/pull/868)
- MNT: allow for exporting of non apogee flights. [#863](https://github.com/RocketPy-Team/RocketPy/pull/863)
- TST: remove remaining files after test session. [#862](https://github.com/RocketPy-Team/RocketPy/pull/862)
- MNT: bumps min python version to 3.10 [#857](https://github.com/RocketPy-Team/RocketPy/pull/857)
- DOC: Update docs dependencies and sub dependencies [#851](https://github.com/RocketPy-Team/RocketPy/pull/851)
- MNT: extract flight data exporters [#845](https://github.com/RocketPy-Team/RocketPy/pull/845)
- ENH: _MotorPrints inheritance - issue #460 [#828](https://github.com/RocketPy-Team/RocketPy/pull/828)
- MNT: fix deprecations and warnings [#829](https://github.com/RocketPy-Team/RocketPy/pull/829)

### Fixed

- BUG: correct encoding for trapezoidal sweep length and angle. [#861](https://github.com/RocketPy-Team/RocketPy/pull/861)
- BUG: Fix no time initialization when passing initial_solution as array to Flight object [#844](https://github.com/RocketPy-Team/RocketPy/pull/844)


## [v1.10.0] - 2025-05-16

### Added

- ENH: Support for ND arithmetic in Function class. [#810] (https://github.com/RocketPy-Team/RocketPy/pull/810)
- ENH: allow users to provide custom samplers [#803](https://github.com/RocketPy-Team/RocketPy/pull/803)
- ENH: Implement Multivariate Rejection Sampling (MRS) [#738] (https://github.com/RocketPy-Team/RocketPy/pull/738)
- ENH: Create a rocketpy file to store flight simulations [#800](https://github.com/RocketPy-Team/RocketPy/pull/800)
- ENH: Support for the RSE file format has been added to the library [#798](https://github.com/RocketPy-Team/RocketPy/pull/798)
- ENH: Introduce Net Thrust with pressure corrections [#789](https://github.com/RocketPy-Team/RocketPy/pull/789)
- ENH: Environment object from EnvironmentAnalysis [#813](https://github.com/RocketPy-Team/RocketPy/pull/813)

### Changed


### Fixed

- BUG: Unecessary Gyroscope Rotation and Wrong Acceleremoter Rotation [#811](https://github.com/RocketPy-Team/RocketPy/pull/811)
- BUG: Fix the handling of reference pressure for older rpy files. [#808](https://github.com/RocketPy-Team/RocketPy/pull/808)
- BUG: Non-overshootable simulations error on time parsing. [#807](https://github.com/RocketPy-Team/RocketPy/pull/807)
- BUG: Wrong Phi Initialization For nose_to_tail Rockets [#809](https://github.com/RocketPy-Team/RocketPy/pull/809)
- BUG: Fix StochasticFlight time_overshoot None bug [#805](https://github.com/RocketPy-Team/RocketPy/pull/805)

## [v1.9.0] - 2025-03-24

### Added

- ENH: Parallel mode for monte-carlo simulations 2 [#768](https://github.com/RocketPy-Team/RocketPy/pull/768)
- DOC: ASTRA Flight Example [#770](https://github.com/RocketPy-Team/RocketPy/pull/770)
- ENH: Add Eccentricity to Stochastic Simulations [#792](https://github.com/RocketPy-Team/RocketPy/pull/792)
- ENH: Introduce the StochasticAirBrakes class [#785](https://github.com/RocketPy-Team/RocketPy/pull/785)

### Changed

- DEP: Remove Pending Deprecations and Add Warnings Where Needed [#794](https://github.com/RocketPy-Team/RocketPy/pull/794)
- DOCS: reshape docs (closes #659) [#781](https://github.com/RocketPy-Team/RocketPy/pull/781)
- MNT: EmptyMotor class inherits from Motor(ABC) [#779](https://github.com/RocketPy-Team/RocketPy/pull/779)

### Fixed

- BUG: do not allow drawing rockets with no aerodynamic surface [#774](https://github.com/RocketPy-Team/RocketPy/pull/774)
- BUG: update flight simulation logic to include burn start time [#778](https://github.com/RocketPy-Team/RocketPy/pull/778)
- BUG: fixes get_instance_attributes for Flight objects containing a Rocket object without rail buttons [#786](https://github.com/RocketPy-Team/RocketPy/pull/786)
- BUG: fixed AGL altitude print for parachutes with lag [#788](https://github.com/RocketPy-Team/RocketPy/pull/788)
- BUG: fix the wind velocity factors usage and better visualization of uniform distributions in Stochastic Classes [#783](https://github.com/RocketPy-Team/RocketPy/pull/783)


## [v1.8.0] - 2025-01-20

To install this version, run `pip install rocketpy==1.8.0`

### Added

- DOC: EREBUS Flight Example [#757](https://github.com/RocketPy-Team/RocketPy/pull/757))
- DOC: Lince Flight Example [#752](https://github.com/RocketPy-Team/RocketPy/pull/752)
- DOC: Andromeda Flight Example [#754](https://github.com/RocketPy-Team/RocketPy/pull/754)
- ENH: create a dataset of pre-registered motors. See #664 [#744](https://github.com/RocketPy-Team/RocketPy/pull/744)
- DOC: add Defiance flight example [#742](https://github.com/RocketPy-Team/RocketPy/pull/742)
- ENH: Allow for Alternative and Custom ODE Solvers. [#748](https://github.com/RocketPy-Team/RocketPy/pull/748)
- ENH: Expansion of Encoders Implementation for Full Flights. [#679](https://github.com/RocketPy-Team/RocketPy/pull/679)

### Changed

- REL: bumps up rocketpy version to 1.8.0 [#762](https://github.com/RocketPy-Team/RocketPy/pull/762)
- ENH: Display more information in MonteCarlo prints and plots [#760](https://github.com/RocketPy-Team/RocketPy/pull/760)
- MNT: move piecewise functions to separate file [#746](https://github.com/RocketPy-Team/RocketPy/pull/746)
- DOC: flight comparison improvements [#755](https://github.com/RocketPy-Team/RocketPy/pull/755)

## [v1.7.1] - 2024-12-07

### Changed

- REL: update version to 1.7.1 in configuration files [#750](https://github.com/RocketPy-Team/RocketPy/pull/750)
- MNT: Refactor Tank's testing Assertion with CAD data. [#678](https://github.com/RocketPy-Team/RocketPy/pull/678)

### Fixed

- BUG: Correctly update atmospheric conditions after changing date and location [#743](https://github.com/RocketPy-Team/RocketPy/pull/743)


## [v1.7.0] - 2024-11-30

### Added

- DOC: GENESIS Flight Example [#734](https://github.com/RocketPy-Team/RocketPy/pull/734)
- DOC: Camoes Flight Example [#733](https://github.com/RocketPy-Team/RocketPy/pull/733)
- ENH: Callback function for collecting additional data from Monte Carlo sims [#702](https://github.com/RocketPy-Team/RocketPy/pull/702)
- ENH: Implement optional plot saving [#597](https://github.com/RocketPy-Team/RocketPy/pull/597)

### Changed

- REL: update version to 1.7.0 in configuration files [#741](https://github.com/RocketPy-Team/RocketPy/pull/741)
- MNT: Place filename save parameter to the end. [#739](https://github.com/RocketPy-Team/RocketPy/pull/739)
- DOC: improvements to developers documentation [#732](https://github.com/RocketPy-Team/RocketPy/pull/732)

### Fixed

- BUG: Allow multiple sets of stochastic fins [#737](https://github.com/RocketPy-Team/RocketPy/pull/737)
- BUG: forecast and reanalysis models - Update ECMWF dictionary values [#736](https://github.com/RocketPy-Team/RocketPy/pull/736)
- BUG: forecast and reanalysis models - move wind_speed to correct position [#735](https://github.com/RocketPy-Team/RocketPy/pull/735)
- BUG: Sideslip Angle and Damping Coefficient Calculation [#729](https://github.com/RocketPy-Team/RocketPy/pull/729)
- DOC: fixed documentation about spherical caps [#728](https://github.com/RocketPy-Team/RocketPy/pull/728)

## [v1.6.2] - 2024-11-08

### Added

- ENH: add structural to total mass ratio for motor and rocket [#713](https://github.com/RocketPy-Team/RocketPy/pull/713)

### Changed

- REL: bumps up rocketpy version to v1.6.2 [#724](https://github.com/RocketPy-Team/RocketPy/pull/724)

### Fixed

- BUG: fix export ellipses to kml function [#712](https://github.com/RocketPy-Team/RocketPy/pull/712)

## [v1.6.1] - 2024-10-10

### Changed

- REL: v1.6.1 [#708](https://github.com/RocketPy-Team/RocketPy/pull/708)
- DEP: deprecate NOAA's RuC sounding [#706](https://github.com/RocketPy-Team/RocketPy/pull/706)

### Fixed

- BUG: Fix Motor Zero Dry Mass Check [#710](https://github.com/RocketPy-Team/RocketPy/pull/710)
- BUG: Fix Environment.max_expected_height for custom atmosphere [#707](https://github.com/RocketPy-Team/RocketPy/pull/707)
- BUG: Initialize _Controller Init Parameters [#703](https://github.com/RocketPy-Team/RocketPy/pull/703)
- BUG: Rail Buttons Not Accepted in Add Surfaces [#701](https://github.com/RocketPy-Team/RocketPy/pull/701)
- BUG: Vector encoding breaks MonteCarlo export. [#704](https://github.com/RocketPy-Team/RocketPy/pull/704)
- BUG: Single Point Functions Can Not Be Defined [#700](https://github.com/RocketPy-Team/RocketPy/pull/700)
- BUG: savetxt Not Accepting lambda Functions [#698](https://github.com/RocketPy-Team/RocketPy/pull/698)

## [v1.6.0] - 2024-09-29

### Added

- REL: v1.6.0 [#697](https://github.com/RocketPy-Team/RocketPy/pull/697)
- ENH: Generic Surfaces and Generic Linear Surfaces [#680](https://github.com/RocketPy-Team/RocketPy/pull/680)
- ENH: Free-Form Fins [#694](https://github.com/RocketPy-Team/RocketPy/pull/694)
- ENH: Expand Polation Options for ND Functions. [#691](https://github.com/RocketPy-Team/RocketPy/pull/691)

## [v1.5.0] - 2024-09-15

### Added

- ENH: Adds Sensors classes [#683](https://github.com/RocketPy-Team/RocketPy/pull/683)
- DOC: Cavour Flight Example [#682](https://github.com/RocketPy-Team/RocketPy/pull/682)
- DOC: Halcyon Flight Example [#681](https://github.com/RocketPy-Team/RocketPy/pull/681)
- ENH: Adds GenericMotor.load_from_eng_file() method [#676](https://github.com/RocketPy-Team/RocketPy/pull/676)
- ENH: Introducing local sensitivity analysis [#575](https://github.com/RocketPy-Team/RocketPy/pull/575)
- ENH: Add STFT function to Function class [#620](https://github.com/RocketPy-Team/RocketPy/pull/620)
- ENH: Rocket Axis Definition [#635](https://github.com/RocketPy-Team/RocketPy/pull/635)

### Changed

- DOC: New Environment class docs pages [#644](https://github.com/RocketPy-Team/RocketPy/pull/644)

### Fixed

- ENH: Fix Orientation Param of Inertial Sensors [#688](https://github.com/RocketPy-Team/RocketPy/pull/688)
- BUG: Zero Mass Flow Rate in Liquid Motors breaks Exhaust Velocity [#677](https://github.com/RocketPy-Team/RocketPy/pull/677)
- DOC: Fix documentation dependencies [#651](https://github.com/RocketPy-Team/RocketPy/pull/651)
- DOC: Fix documentation warnings [#645](https://github.com/RocketPy-Team/RocketPy/pull/645)
- BUG: Rotational EOMs Not Relative To CDM [#674](https://github.com/RocketPy-Team/RocketPy/pull/674)
- BUG: Pressure ISA Extrapolation as "linear" [#675](https://github.com/RocketPy-Team/RocketPy/pull/675)
- BUG: fix the Frequency Response plot of Flight class [#653](https://github.com/RocketPy-Team/RocketPy/pull/653)

## [v1.4.3] - 2024-09-11

You can install this version by running `pip install rocketpy==1.4.3`

### Changed

- REL: Bump versioning to RocketPy v1.4.3 [#687](https://github.com/RocketPy-Team/RocketPy/pull/687)

### Fixed

- BUG: Rollback Prandtl-Glauert corrections for Tail and Nose. [#685](https://github.com/RocketPy-Team/RocketPy/pull/685)

## [v1.4.2] - 2024-08-03

You can install this version by running `pip install rocketpy==1.4.2`

### Changed

- REL: Bump versioning to RocketPy v1.4.2 [#648](https://github.com/RocketPy-Team/RocketPy/pull/648)
- ENH: Adding rocket radius to RailButtons class [#643](https://github.com/RocketPy-Team/RocketPy/pull/643)

### Fixed

- BUG: Time Node Merge Not Including Controllers [#647](https://github.com/RocketPy-Team/RocketPy/pull/647)

## [v1.4.1] - 2024-07-20

You can install this version by running `pip install rocketpy==1.4.1`

### Changed

- REL: Bumps rocketpy version to 1.4.1 [#646](https://github.com/RocketPy-Team/RocketPy/pull/646)
- ENH: Insert apogee state into solution list during flight simulation [#638](https://github.com/RocketPy-Team/RocketPy/pull/638)
- MNT: Refactor AeroSurfaces [#634](https://github.com/RocketPy-Team/RocketPy/pull/634)
- ENH: Environment class major refactor may 2024 [#605](https://github.com/RocketPy-Team/RocketPy/pull/605)
- MNT: Refactors the code to adopt flake8 [#631](https://github.com/RocketPy-Team/RocketPy/pull/631)
- MNT: Refactors the code to adopt pylint [#621](https://github.com/RocketPy-Team/RocketPy/pull/621)

## [1.4.0] - 2024-07-06

You can install this version by running `pip install rocketpy==1.4.0`

### Added

- DOC: Adding testing guidelines for RocketPy. [#626](https://github.com/RocketPy-Team/RocketPy/pull/626)
- ENH: CP and Thrust Eccentricity Effects Generate Roll Moment [#617](https://github.com/RocketPy-Team/RocketPy/pull/617)
- ENH: Add Prandtl-Gauss transformation to NoseCone and Tail [#609](https://github.com/RocketPy-Team/RocketPy/pull/609)
- ENH: Implement power series nose cones [#603](https://github.com/RocketPy-Team/RocketPy/pull/603)

### Changed

- ENH: Eliminating multiple plots for inertia components [#566](https://github.com/RocketPy-Team/RocketPy/pull/566)
- MNT: Fix warnings in test suite and adds support for numpy 2.0 [#623](https://github.com/RocketPy-Team/RocketPy/pull/623)
- MNT: bump minimum Python version to 3.9. [#624](https://github.com/RocketPy-Team/RocketPy/pull/624)
- DOC: Change rocketpy Landing Page to Standard Code docs [#584](https://github.com/RocketPy-Team/RocketPy/pull/584)

## [1.3.0.post1] - 2024-06-02

You can install this version by running `pip install rocketpy==1.3.0.post1`

### Fixed

- BUG: pyproject.toml Main Module Finding. [#616](https://github.com/RocketPy-Team/RocketPy/pull/616)

## [1.3.0] - 2024-06-01

You can install this version by running `pip install rocketpy==1.3.0`

### Added

- DOC: Adds prometheus data, Spaceport America 2022 [#601](https://github.com/RocketPy-Team/RocketPy/pull/601)
- ENH: Pre-calculate attributes in Rocket class [#595](https://github.com/RocketPy-Team/RocketPy/pull/595)
- ENH: Complex step differentiation [#594](https://github.com/RocketPy-Team/RocketPy/pull/594)
- ENH: Exponential backoff decorator (fix #449) [#588](https://github.com/RocketPy-Team/RocketPy/pull/588)
- ENH: Function Validation Rework & Swap `np.searchsorted` to `bisect_left` [#582](https://github.com/RocketPy-Team/RocketPy/pull/582)
- ENH: Add new stability margin properties to Flight class [#572](https://github.com/RocketPy-Team/RocketPy/pull/572)
- ENH: adds `Function.remove_outliers` method [#554](https://github.com/RocketPy-Team/RocketPy/pull/554)

### Changed

- REL: Bump versioning to RocketPy v1.3.0 [#614](https://github.com/RocketPy-Team/RocketPy/pull/614)
- ENH: Adds StochasticModel.visualize_attributes() method [#612](https://github.com/RocketPy-Team/RocketPy/pull/612)
- DOC: Monte carlo documentation updates [#607](https://github.com/RocketPy-Team/RocketPy/pull/607)
- MNT: refactor u_dot parachute method [#596](https://github.com/RocketPy-Team/RocketPy/pull/596)
- BLD: Change setup.py to pyproject.toml [#589](https://github.com/RocketPy-Team/RocketPy/pull/589)
- DEP: delete deprecated rocketpy.tools.cached_property [#587](https://github.com/RocketPy-Team/RocketPy/pull/587)
- ENH: Flight simulation speed up [#581](https://github.com/RocketPy-Team/RocketPy/pull/581)
- MNT: Modularize Rocket Draw [#580](https://github.com/RocketPy-Team/RocketPy/pull/580)
- DOC: Improvements of Environment docstring phrasing [#565](https://github.com/RocketPy-Team/RocketPy/pull/565)
- MNT: Refactor flight prints module [#579](https://github.com/RocketPy-Team/RocketPy/pull/579)
- DOC: Convert CompareFlights example notebooks to .rst files [#576](https://github.com/RocketPy-Team/RocketPy/pull/576)
- MNT: Refactor inertia calculations using parallel axis theorem [#573](https://github.com/RocketPy-Team/RocketPy/pull/573)
- ENH: Optional argument to show the plot in Function.compare_plots [#563](https://github.com/RocketPy-Team/RocketPy/pull/563)

### Fixed

- BUG: Fixes StochasticNoseCone powerseries issue #838 [#839](https://github.com/RocketPy-Team/RocketPy/pull/839)
- MNT: Alter PYPI classifier naming. [#615](https://github.com/RocketPy-Team/RocketPy/pull/615)
- DOC: Solve Dependencies Conflicts and pyproject build [#613](https://github.com/RocketPy-Team/RocketPy/pull/613)
- BUG: Fixes nose cone bluffness issue #610 [#611](https://github.com/RocketPy-Team/RocketPy/pull/611)
- BUG: plot drag curves when function source is callable [#599](https://github.com/RocketPy-Team/RocketPy/pull/599)
- BUG: Fix minor type hinting problems [#598](https://github.com/RocketPy-Team/RocketPy/pull/598)
- BUG: Optional Dependencies Naming in pyproject.toml. [#592](https://github.com/RocketPy-Team/RocketPy/pull/592)
- BUG: Swap rocket.total_mass.differentiate for motor.total_mass_flow rate [#585](https://github.com/RocketPy-Team/RocketPy/pull/585)
- BUG: export_eng 'Motor' method would not work for liquid motors. [#559](https://github.com/RocketPy-Team/RocketPy/pull/559)

## [v1.2.2] - 2024-03-22

You can install this version by running `pip install rocketpy==1.2.2`

- BUG: wrong rocket mass in parachute u dot method [#569](https://github.com/RocketPy-Team/RocketPy/pull/569)

## [v1.2.1] - 2024-02-22

You can install this version by running `pip install rocketpy==1.2.1`

### Fixed

- BUG: Add reference area factor correction to aero surfaces (solves #557) [#558](https://github.com/RocketPy-Team/RocketPy/pull/558)

## [v1.2.0] - 2024-02-12

You can install this version by running `pip install rocketpy==1.2.0`

### Added

- ENH: Function Support for CSV Header Inputs [#542](https://github.com/RocketPy-Team/RocketPy/pull/542)
- ENH: Argument for Optional Mutation on Function Discretize [#519](https://github.com/RocketPy-Team/RocketPy/pull/519)
- ENH: Add of a line for saving the filtered dataset [#518](https://github.com/RocketPy-Team/RocketPy/pull/518)
- ENH: Shepard Optimized Interpolation - Multiple Inputs Support [#515](https://github.com/RocketPy-Team/RocketPy/pull/515)
- ENH: adds new Function.savetxt method [#514](https://github.com/RocketPy-Team/RocketPy/pull/514)
- DOC: add juno3 flight example [#513](https://github.com/RocketPy-Team/RocketPy/pull/513)
- ENH: add Function.low_pass_filter method [#508](https://github.com/RocketPy-Team/RocketPy/pull/508)
- ENH: Air Brakes [#426](https://github.com/RocketPy-Team/RocketPy/pull/426)
-

### Changed

- MNT: Final refactor before v1.2 [#553](https://github.com/RocketPy-Team/RocketPy/pull/553)
- ENH: Plotting both power on and off drag curves in a single plot [#547](https://github.com/RocketPy-Team/RocketPy/pull/547)
- DOC: Replacing git clone command with curl in notebooks. [#544](https://github.com/RocketPy-Team/RocketPy/pull/544)
- DOC: Installing imageio library on dispersion analysis notebook [#540](https://github.com/RocketPy-Team/RocketPy/pull/540)
- MNT: improve the low pass filter and document an example [#538](https://github.com/RocketPy-Team/RocketPy/pull/538)
- MNT: Encapsulate quaternion conversions [#537](https://github.com/RocketPy-Team/RocketPy/pull/537)
- ENH: Precalculate Barometric Height [#511](https://github.com/RocketPy-Team/RocketPy/pull/511)
- ENH: optimize get_value_opt in class Function [#501](https://github.com/RocketPy-Team/RocketPy/pull/501)
- DOC: Update Header Related Docs [#495](https://github.com/RocketPy-Team/RocketPy/pull/495)
- ENH: Function Reverse Arithmetic Priority [#488](https://github.com/RocketPy-Team/RocketPy/pull/488)
- MNT: Add repr method to Parachute class [#490](https://github.com/RocketPy-Team/RocketPy/pull/490)

### Fixed

- BUG: Update flight trajectory plot axes limits [#552](https://github.com/RocketPy-Team/RocketPy/pull/552)
- BUG: fix `get_controller_observed_variables` in the air brakes examples [#551](https://github.com/RocketPy-Team/RocketPy/pull/551)
- MNT: small fixes before v1.2 [#550](https://github.com/RocketPy-Team/RocketPy/pull/550)
- BUG: Elliptical Fins Draw [#548](https://github.com/RocketPy-Team/RocketPy/pull/548)
- BUG: 3D trajectory plot not labeling axes [#533](https://github.com/RocketPy-Team/RocketPy/pull/533)
- FIX: EmptyMotor is breaking the Rocket.draw() method [#516](https://github.com/RocketPy-Team/RocketPy/pull/516)
- BUG: fin_flutter_analysis doesn't find any fin set [#510](https://github.com/RocketPy-Team/RocketPy/pull/510)
- ENH: Parachute trigger doesn't work if "Apogee" is used instead of "apogee" [#489](https://github.com/RocketPy-Team/RocketPy/pull/489)


## [v1.1.5] - 2024-01-21

You can install this version by running `pip install rocketpy==1.1.5`

### Fixed

- BUG: Parachute Pressures not being Set before All Info. [#534](https://github.com/RocketPy-Team/RocketPy/pull/534)
- BUG: Invalid Arguments on Two Dimensional Discretize. [#521](https://github.com/RocketPy-Team/RocketPy/pull/521)

## [v1.1.4] - 2023-12-09

You can install this version by running `pip install rocketpy==1.1.4`

### Fixed

- FIX: changes Generic Motor exhaust velocity to cached property [#497](https://github.com/RocketPy-Team/RocketPy/pull/497)
- DOC: Change from % to ! in the first cell to run properly in Colab. [#496](https://github.com/RocketPy-Team/RocketPy/pull/496)

## [v1.1.3] - 2023-11-29

You can install this version by running `pip install rocketpy==1.1.3`

### Fixed

- FIX: Never ending Flight simulations when using a GenericMotor [#497](https://github.com/RocketPy-Team/RocketPy/pull/497)
- FIX: Broken Function.get_value_opt for N-Dimensional Functions [#492](https://github.com/RocketPy-Team/RocketPy/pull/492)

## [v1.1.2] - 2023-11-27

You can install this version by running `pip install rocketpy==1.1.2`

### Fixed

- BUG: Function breaks if a header is present in the csv file [#485](https://github.com/RocketPy-Team/RocketPy/pull/485)

## [v1.1.1] - 2023-11-24

You can install this version by running `pip install rocketpy==1.1.1`

### Added

- ENH: Prevent out of bounds Tanks from Instantiation #484 [#484](https://github.com/RocketPy-Team/RocketPy/pull/484)
- DOC: Added this changelog file [#472](https://github.com/RocketPy-Team/RocketPy/pull/472)

### Fixed

- HOTFIX: Tanks Overfill not Being Detected [#479](https://github.com/RocketPy-Team/RocketPy/pull/479)
- HOTFIX: 2D .CSV Function and missing set_get_value_opt call [#478](https://github.com/RocketPy-Team/RocketPy/pull/478)
- HOTFIX: Negative Static Margin [#476](https://github.com/RocketPy-Team/RocketPy/pull/476)

## [v1.1.0] - 2023-11-19

You can install this version by running `pip install rocketpy==1.1.0`

### Added

- DOC: first simulation all_info [#466](https://github.com/RocketPy-Team/RocketPy/pull/466)
- DOC: Documentation for Function Class Usage [#465](https://github.com/RocketPy-Team/RocketPy/pull/465)
- DOC: add documentation for flight data export. [#464](https://github.com/RocketPy-Team/RocketPy/pull/464)
- ENH: Add mass_flow_rate() to GenericMotor class [#459](https://github.com/RocketPy-Team/RocketPy/pull/459)
- DOC: Add documentation on how to build the docs [#452](https://github.com/RocketPy-Team/RocketPy/pull/452)
- ENH: draw motors [#436](https://github.com/RocketPy-Team/RocketPy/pull/436)
- ENH: new Flight.get_solution_at_time() method [#441](https://github.com/RocketPy-Team/RocketPy/pull/441)
- ENH: rocket drawing [419](https://github.com/RocketPy-Team/RocketPy/pull/419)
- ENH: Adding Stability Margin with Mach dependency [#377](https://github.com/RocketPy-Team/RocketPy/pull/377)

### Changed

- ENH: Clean Plots and Prints sub packages init files [#457](https://github.com/RocketPy-Team/RocketPy/pull/457)
- ENH: Add \_MotorPlots Inheritance to Motor Plots Classes [#456](https://github.com/RocketPy-Team/RocketPy/pull/456)
- ENH: Spherical Caps Included in Total Length [#455](https://github.com/RocketPy-Team/RocketPy/pull/455)
  - Important: This changes behavior of `TankGeometry.add_spherical_caps()`
- DOC: improve mass and inertia docs [#445](https://github.com/RocketPy-Team/RocketPy/pull/445)
- DOC: organize flight examples folder [#429](https://github.com/RocketPy-Team/RocketPy/pull/429)

### Fixed

- BUG: Fix find_input() Function to Return a Single Value [#471](https://github.com/RocketPy-Team/RocketPy/pull/471)
- MNT: Refactor exhaust velocity calculation to avoid ZeroDivisionError [#470](https://github.com/RocketPy-Team/RocketPy/pull/470)
- DOC: refactor dispersion analysis notebook [#463](https://github.com/RocketPy-Team/RocketPy/pull/463)
- DOC: fix positions and coordinate system documentation page [#454](https://github.com/RocketPy-Team/RocketPy/pull/454)
- BUG: User input checks added for Function class [#451](https://github.com/RocketPy-Team/RocketPy/pull/451)
- MNT: fix env plots legends [#440](https://github.com/RocketPy-Team/RocketPy/pull/440)
- BUG: flight.prints.max_values() fails when launching an EmptyMotor [#438](https://github.com/RocketPy-Team/RocketPy/pull/438)
- MNT: Fix env plots max heights [#433](https://github.com/RocketPy-Team/RocketPy/pull/433)
- BUG: Maintaining Extrapolation when Adding Discrete Functions with Constants [#432](https://github.com/RocketPy-Team/RocketPy/pull/432)


# [v1.0.1] - 2023-10-07

You can install this version by running `pip install rocketpy==1.0.1`

### Fixed

- BUG: Remove NoseCone Warning [#428](https://github.com/RocketPy-Team/RocketPy/pull/428)
- BUG: motor coordinates [#423](https://github.com/RocketPy-Team/RocketPy/pull/423)
