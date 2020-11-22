Device & Control 
====================

描述装置运行状态和控制过程

数据来源：
   - 装置工程位形参数
   - 控制运行设计方案
   - 实际控制运行参数



控制系统
-----------
.. toctree::
   :maxdepth: 2

   ../_apidoc/control.SDN
   ../_apidoc/control.Controllers
   ../_apidoc/control.PulseSchedule


装置位形，磁场线圈
--------------------
.. toctree::
   :maxdepth: 2

   ../_apidoc/Wall
   ../_apidoc/PFActive
   ../_apidoc/TF
   ../_apidoc/Magnetic
   ../_apidoc/control.IronCore
   ../_apidoc/control.CoilsNonAxisymmetric


加热、驱动和加料
-----------------

.. toctree::
   :maxdepth: 2

   ../_apidoc/control.ECLaunchers
   ../_apidoc/control.EMCoupling
   ../_apidoc/control.GasInjection
   ../_apidoc/control.ICAntennas
   ../_apidoc/control.LHAntennas
   ../_apidoc/control.NBI
   ../_apidoc/control.Pellets



.. amns_data                Atomic, molecular, nuclear and surface physics data. Each occurrence contains the data for a given element (nuclear charge), describing various physical processes. For each process, data tables are organized by charge states. The coordinate system used by the data tables is described under the coordinate_system node.    2
.. barometry                Pressure measurements in the vacuum vessel. NB will need to change the type of the pressure node to signal_1d when moving to the new LL.    2
.. bolometer                Bolometer diagnostic    2
.. bremsstrahlung_visible   Diagnostic for measuring the bremsstrahlung from thermal particules in the visible light range, in view of determining the effective charge of the plasma.    2
.. calorimetry              Calometry measurements on various tokamak subsystems    2
.. camera_ir                Infrared camera for monitoring of Plasma Facing Components    10
.. camera_visible           Camera in the visible light range    20
.. charge_exchange          Charge exchange spectroscopy diagnostic    4
.. coils_non_axisymmetric   Non axisymmetric active coils system (e.g. ELM control coils, error field correction coils, ...)    5
.. controllers              Feedback and feedforward controllers    2
.. core_instant_changes     Instant changes of the radial core plasma profiles due to pellet, MHD, ...    3
.. core_profiles            Core plasma radial profiles    15
.. core_sources             Core plasma thermal source terms (for the transport equations of the thermal species). Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)    15
.. core_transport           Core plasma transport of particles, energy, momentum and poloidal flux. The transport of particles, energy and momentum is described by diffusion coefficients, D, and convection velocities, v. These are defined by the total fluxes of particles, energy and momentum, across a flux surface given by : V' [-D Y' <|grad(rho_tor_norm)|^2gt; + v Y <|grad(rho_tor_norm)|>], where Y represents the particles, energy and momentum density, respectively, while V is the volume inside a flux surface, the primes denote derivatives with respect to rho_tor_norm and < X > is the flux surface average of a quantity X. This formulation remains valid when changing simultaneously rho_tor_norm into rho_tor in the gradient terms and in the derivatives denoted by the prime. The average flux stored in the IDS as sibling of D and v is the total flux described above divided by the flux surface area V' <|grad(rho_tor_norm)|>. Note that the energy flux includes the energy transported by the particle flux.    10
.. cryostat                 Description of the cryostat surrounding the machine (if any)    1
.. dataset_description      General description of the dataset (collection of all IDSs within the given database entry). Main description text to be put in ids_properties/comment    1
.. dataset_fair             FAIR metadata related to the dataset, providing inforrmation on licensing, annotations, references using this dataset, versioning and validity, provenance. This IDS is using Dublin Core metadata standard whenever possible    1
.. disruption               Description of physics quantities of specific interest during a disruption, in particular halo currents, etc ...    1
.. distribution_sources     Sources of particles for input to kinetic equations, e.g. Fokker-Planck calculation. The sources could originate from e.g. NBI or fusion reactions.    4
.. distributions            Distribution function(s) of one or many particle species. This structure is specifically designed to handle non-Maxwellian distribution function generated during heating and current drive, typically solved using a Fokker-Planck calculation perturbed by a heating scheme (e.g. IC, EC, LH, NBI, or alpha heating) and then relaxed by Coloumb collisions.    8
.. ec_launchers             Launchers for heating and current drive in the electron cyclotron (EC) frequencies.    2
.. ece                      Electron cyclotron emission diagnostic    3
.. edge_profiles            Edge plasma profiles (includes the scrape-off layer and possibly part of the confined plasma)    10
.. edge_sources             Edge plasma sources. Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)    10
.. edge_transport           Edge plasma transport. Energy terms correspond to the full kinetic energy equation (i.e. the energy flux takes into account the energy transported by the particle flux)    10
.. em_coupling              Description of the axisymmetric mutual electromagnetics; does not include non-axisymmetric coil systems; the convention is Quantity_Sensor_Source    3
.. equilibrium              Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.    3
.. gas_injection            Gas injection by a system of pipes and valves    2
.. gyrokinetics             Description of a gyrokinetic simulation (delta-f, flux-tube). All quantities within this IDS are normalised (apart from time), thus independent of rhostar, consistently with the local approximation and a spectral representation is assumed in the perpendicular plane (i.e. homogeneous turbulence). All quantities are given in the laboratory frame, except the moments of the perturbed distribution function which are given in the rotating frame.    1
.. hard_x_rays              Hard X-rays tomography diagnostic    2
.. ic_antennas              Antenna systems for heating and current drive in the ion cylcotron (IC) frequencies.    2
.. interferometer           Interferometer diagnostic    5
.. iron_core                Iron core description    1
.. langmuir_probes          Langmuir probes    3
.. lh_antennas              Antenna systems for heating and current drive in the Lower Hybrid (LH) frequencies. In the definitions below, the front (or mouth) of the antenna refers to the plasma facing side of the antenna, while the back refers to the waveguides connected side of the antenna (towards the RF generators).    2
.. magnetics                Magnetic diagnostics for equilibrium identification and plasma shape control.    3
.. mhd                      Magnetohydrodynamic activity, description of perturbed fields and profiles using the Generic Grid Description.    2
.. mhd_linear               Magnetohydronamic linear stability    5
.. mse                      Motional Stark Effect diagnostic    2
.. nbi                      Neutral Beam Injection systems and description of the fast neutrals that arrive into the torus    3
.. neutron_diagnostic       Neutron diagnostic such as DNFM, NFM or MFC    3
.. ntms                     Description of neoclassical tearing modes    1
.. numerics                 Numeric parameters passed as argument to a component or a workflow. Most quantities are dynamic in this IDS in order to record the history of the numerics parameters at each execution of the component or workflow (so one time index = one execution of the component). Provide as input to the component a single time slice from this IDS containing the relevant parameters.    10
.. pellets                  Description of pellets launched into the plasma    2
.. pf_active                Description of the axisymmetric active poloidal field (PF) coils and supplies; includes the limits of these systems; includes the forces on them; does not include non-axisymmetric coil systems    3
.. pf_passive               Description of the axisymmetric passive conductors, currents flowing in them    3
.. polarimeter              Polarimeter diagnostic    2
.. pulse_schedule           Description of Pulse Schedule, described by subsystems waveform references and an enveloppe around them. The controllers, pulse schedule and SDN are defined in separate IDSs. All names and identifiers of subsystems appearing in the pulse_schedule must be identical to those used in the IDSs describing the related subsystems.    1
.. radiation                Radiation emitted by the plasma and neutrals    2
.. reflectometer_profile    Profile reflectometer diagnostic. Multiple reflectometers are considered as independent diagnostics to be handled with different occurrence numbers    2
.. sawteeth                 Description of sawtooth events. This IDS must be used in homogeneous_time = 1 mode    2
.. sdn                      Description of the Synchronous Data Network parameters and the signals on it    3
.. soft_x_rays              Soft X-rays tomography diagnostic    1
.. spectrometer_mass        Mass spectrometer diagnostic    4
.. spectrometer_uv          Spectrometer in uv light range diagnostic    2
.. spectrometer_visible     Spectrometer in visible light range diagnostic    1
.. spectrometer_x_ray_crystal    X-crystal spectrometer diagnostic    2
.. summary                  Summary of physics quantities from a simulation or an experiment. Dynamic quantities are either taken at given time slices (indicated in the "time" vector) or time-averaged over an interval (in such case the "time_width" of the interval is indicated and the "time" vector represents the end of each time interval).    1
.. temporary                Storage of undeclared data model components    6
.. thomson_scattering       Thomson scattering diagnostic    3
.. tf                       Toroidal field coils    3
.. transport_solver_numerics    Numerical quantities used by transport solvers and convergence details    6
.. turbulence               Description of plasma turbulence    2
.. wall                     Description of the torus wall and its interaction with the plasma    5
.. waves                    RF wave propagation and deposition. Note that current estimates in this IDS are a priori not taking into account synergies between multiple sources (a convergence loop with Fokker-Planck calculations is required to account for such synergies)
