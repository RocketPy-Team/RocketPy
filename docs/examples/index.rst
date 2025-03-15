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

   import plotly.graph_objects as go

   results = {
      # "Name (Year)": (simulated, measured) m
      # - use 2 decimal places
      # - sort by year and then by name
      "Valetudo (2019)": (825.39, 860),
      "Bella Lui (2020)": (460.50, 458.97),
      "NDRT (2020)": (1296.77, 1316.75),
      "Andromeda (2022)": (3614.95,3415),
      "Astra (2022)": (3043.07, 3250), 
      "Erebus (2022)": (2750.90,3020),
      "Prometheus (2022)": (4190.05, 3898.37),
      "Cavour (2023)": (2818.90, 2789),
      "Camoes (2023)": (3003.28, 3015),
      "Juno III (2023)": (3026.05, 3213),
      "Genesis (2023)": (3076.45, 2916),
      "Halcyon (2023)": (3212.78, 3450),
      "Lince (2023)": (3284.12, 3587),
      "Defiance (2024)": (9238.01, 9308.32),
   }

   max_apogee = 10000

   # Extract data
   simulated = [sim for sim, meas in results.values()]
   measured = [meas for sim, meas in results.values()]
   labels = list(results.keys())

   # Create the plot
   fig = go.Figure()

   # Add the x = y line
   fig.add_trace(go.Scatter(
      x=[0, max_apogee],
      y=[0, max_apogee],
      mode='lines',
      line=dict(dash='dash', color='black'),
      showlegend=False
   ))

   # Add scatter plot
   fig.add_trace(go.Scatter(
      x=simulated,
      y=measured,
      mode='markers',
      text=labels,
      textposition='top center',
      marker=dict(size=10),
      showlegend=False
   ))

   # Set titles and labels
   fig.update_layout(
      title="Simulated x Measured Apogee",
      xaxis_title="Simulated Apogee (m)",
      yaxis_title="Measured Apogee (m)",
      xaxis=dict(range=[0, max_apogee]),
      yaxis=dict(range=[0, max_apogee]),
      width=650,
      height=650
   )

   fig.show()

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
   andromeda_flight_sim.ipynb
   astra_flight_sim.ipynb
   prometheus_2022_flight_sim.ipynb
   erebus_flight_sim.ipynb
   halcyon_flight_sim.ipynb
   cavour_flight_sim.ipynb
   genesis_flight_sim.ipynb
   camoes_flight_sim.ipynb
   lince_flight_sim.ipynb
   defiance_flight_sim.ipynb


