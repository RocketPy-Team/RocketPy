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

<!-- These are the changes that were not release yet, please add them correctly.
  Attention: The newest changes should be on top -->

### Added

- ENH: Pre-calculate attributes in Rocket class [#595](https://github.com/RocketPy-Team/RocketPy/pull/595)
- ENH: Complex step differentiation [#594](https://github.com/RocketPy-Team/RocketPy/pull/594)
- ENH: Exponential backoff decorator (fix #449) [#588](https://github.com/RocketPy-Team/RocketPy/pull/588)
- ENH: Function Validation Rework & Swap `np.searchsorted` to `bisect_left` [#582](https://github.com/RocketPy-Team/RocketPy/pull/582)
- ENH: Add new stability margin properties to Flight class [#572](https://github.com/RocketPy-Team/RocketPy/pull/572)
- ENH: adds `Function.remove_outliers` method [#554](https://github.com/RocketPy-Team/RocketPy/pull/554)

### Changed

- BLD: Change setup.py to pyproject.toml [#589](https://github.com/RocketPy-Team/RocketPy/pull/589)
- DEP: delete deprecated rocketpy.tools.cached_property [#587](https://github.com/RocketPy-Team/RocketPy/pull/587)
- ENH: Flight simulation speed up [#581] (https://github.com/RocketPy-Team/RocketPy/pull/581) 
- MNT: Modularize Rocket Draw [#580](https://github.com/RocketPy-Team/RocketPy/pull/580)
- DOC: Improvements of Environment docstring phrasing [#565](https://github.com/RocketPy-Team/RocketPy/pull/565)
- MNT: Refactor flight prints module [#579](https://github.com/RocketPy-Team/RocketPy/pull/579)
- DOC: Convert CompareFlights example notebooks to .rst files [#576](https://github.com/RocketPy-Team/RocketPy/pull/576)
- MNT: Refactor inertia calculations using parallel axis theorem [#573] (https://github.com/RocketPy-Team/RocketPy/pull/573)
- ENH: Optional argument to show the plot in Function.compare_plots [#563](https://github.com/RocketPy-Team/RocketPy/pull/563)

### Fixed

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
