Flights simulated with RocketPy
===============================

RocketPy has been used to simulate many flights, from small amateur rockets to
large professional ones.
This section contains some of the most interesting
results obtained with RocketPy.
The following plot shows the comparison between the simulated and measured
apogee of some rockets.

.. jupyter-execute::
   :hide-code:

   import matplotlib.pyplot as plt

   results = {
      # "Name (Year)": (simulated, measured) m
      # - use 2 decimal places
      # - sort by year and then by name
      "Valetudo (2019)": (825.39, 860),
      "Bella Lui (2020)": (460.50, 458.97),
      "NDRT (2020)": (1296.77, 1316.75),
      "Prometheus (2022)": (4190.05, 3898.37),
      "Cavour (2023)": (2818.90, 2789),
      "Genesis (2023)": (3076.45, 2916),
      "Camoes (2023)": (3003.28, 3015),
      "Juno III (2023)": (3026.05, 3213),
      "Halcyon (2023)": (3212.78, 3450),
      "Defiance Mk.IV (2024)": (9238.01, 9308.32),
   }

   max_apogee = 10000

   # Extract data
   simulated = [sim for sim, meas in results.values()]
   measured = [meas for sim, meas in results.values()]
   labels = list(results.keys())

   # Create the plot
   fig, ax = plt.subplots(figsize=(9, 9))
   ax.scatter(simulated, measured)
   ax.grid(True, alpha=0.3)

   # Add the x = y line
   ax.plot([0, max_apogee], [0, max_apogee], linestyle='--', color='black', alpha=0.6)

   # Add text labels
   for i, label in enumerate(labels):
      ax.text(simulated[i], measured[i], label, ha='center', va='bottom', fontsize=8)

   # Set titles and labels
   ax.set_title("Simulated x Measured Apogee")
   ax.set_xlabel("Simulated Apogee (m)")
   ax.set_ylabel("Measured Apogee (m)")

   # Set aspect ratio to 1:1
   ax.set_aspect('equal', adjustable='box')
   ax.set_xlim(0, max_apogee)
   ax.set_ylim(0, max_apogee)

   plt.show()

In the next sections you will find the simulations of the rockets listed above.

.. note::

   If you want to see your rocket here, please contact the maintainers! \
   We would love to include your rocket in the examples.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   bella_lui_flight_sim.ipynb
   ndrt_2020_flight_sim.ipynb
   valetudo_flight_sim.ipynb
   SEB_liquid_motor.ipynb
   juno3_flight_sim.ipynb
   prometheus_2022_flight_sim.ipynb
   halcyon_flight_sim.ipynb
   cavour_flight_sim.ipynb
   genesis_flight_sim.ipynb
   camoes_flight_sim.ipynb
   defiance_flight_sim.ipynb

